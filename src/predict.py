"""Single-image inference for document forgery detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import UnifiedForgeryModel

try:
    import easyocr
    _OCR_READER = None  # lazy init
except Exception:
    easyocr = None
    _OCR_READER = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _load_rgb(path: Path, size: int = 224) -> np.ndarray:
    # Loads the input image, converts it to RGB, and resizes it
    # to the model input size expected during inference.
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size))


def _forensic_stack(rgb: np.ndarray) -> np.ndarray:
    """Generate ELA + SRM + FFT maps on-the-fly from an RGB image."""
    from PIL import Image, ImageChops, ImageEnhance
    import tempfile, os

    pil = Image.fromarray(rgb)

    # ELA:
    # Recompress the image as JPEG and measure the pixel-wise difference
    # between the original and recompressed versions to reveal compression artifacts.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    pil.save(tmp_path, "JPEG", quality=90)
    recomp = Image.open(tmp_path).convert("RGB")
    os.unlink(tmp_path)
    diff = ImageChops.difference(pil, recomp)
    extrema = diff.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    ela_img = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)
    ela = np.array(ela_img.convert("L"), dtype=np.float32) / 255.0

    # SRM:
    # Applies a residual high-pass kernel to highlight subtle noise inconsistencies
    # that may indicate tampering or synthesis.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    srm_kernel = np.array(
        [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
        dtype=np.float32,
    )
    srm = cv2.filter2D(gray, -1, srm_kernel)
    srm = cv2.normalize(np.abs(srm), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # FFT:
    # Converts the image into frequency space to expose unusual periodic
    # or generation-related artifacts that are less visible in pixel space.
    fft = np.fft.fftshift(np.fft.fft2(gray))
    fft_mag = 20 * np.log(np.abs(fft) + 1.0)
    fft_map = cv2.normalize(fft_mag, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Returns the three forensic maps stacked as a 3-channel tensor-like array.
    return np.stack([ela, srm, fft_map], axis=0)  # (3, H, W)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def _gradcam_vit(
    model: UnifiedForgeryModel,
    rgb_t: torch.Tensor,
    fore_t: torch.Tensor,
    target_class: int,
    head: str = "stage1",
) -> np.ndarray:
    # Stores forward activations and backward gradients from the last ViT block
    # so a Grad-CAM-like heatmap can be computed for the chosen target class.
    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["feat"] = output

    def bwd_hook(_, __, grad_output):
        gradients["feat"] = grad_output[0]

    # Register hooks on the final transformer block of the RGB branch
    # to capture the information needed for visualization.
    last_block = model.rgb_branch.blocks[-1]
    fwd = last_block.register_forward_hook(fwd_hook)
    bwd = last_block.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    rgb_t = rgb_t.requires_grad_(True)
    out = model(rgb_t, fore_t)

    # Choose which classifier head to explain.
    if head == "stage2":
        logits = out["stage2_logits"]
    else:
        logits = out["stage1_logits"]

    # Backpropagate only the selected target class score.
    logits[0, target_class].backward()

    # Remove hooks immediately after use to avoid side effects in later calls.
    fwd.remove()
    bwd.remove()

    acts = activations["feat"]
    grads = gradients["feat"]

    # Ignore the CLS token and use only patch tokens for spatial localization.
    patch_acts = acts[0, 1:]
    patch_grads = grads[0, 1:]

    # Average gradients across channels to obtain token importance weights,
    # then compute a weighted sum over patch activations.
    weights = patch_grads.mean(dim=-1, keepdim=True)
    cam = (weights * patch_acts).sum(dim=-1)
    cam = F.relu(cam)

    # Reshape the 1D token sequence back into a 2D map,
    # resize it to image size, and normalize to 8-bit grayscale.
    side = int(cam.shape[0] ** 0.5)
    cam = cam.reshape(side, side).detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cam

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def _ocr_caption(heatmap_gray: np.ndarray, rgb: np.ndarray) -> str:
    """Run OCR on the hottest region to name the altered field."""
    # If EasyOCR is unavailable, fall back to a generic caption.
    if easyocr is None:
        return "Forged — Edited. A region appears altered."

    global _OCR_READER
    if _OCR_READER is None:
        # Lazily initialize OCR only when first needed.
        _OCR_READER = easyocr.Reader(["en"], gpu=False)

    # Build a mask covering the most suspicious region in the heatmap.
    # First try a strong threshold; if that fails, fall back to a relative threshold.
    mask = (heatmap_gray > 180).astype(np.uint8)
    if mask.sum() == 0:
        mask = (heatmap_gray > heatmap_gray.max() * 0.7).astype(np.uint8)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return "Forged — Edited. A region appears altered."

    # Crop the highlighted region with a small margin
    # and run OCR on that crop to identify likely field text.
    y1, y2 = max(ys.min() - 10, 0), min(ys.max() + 10, rgb.shape[0])
    x1, x2 = max(xs.min() - 10, 0), min(xs.max() + 10, rgb.shape[1])
    crop = rgb[y1:y2, x1:x2]

    results = _OCR_READER.readtext(crop)
    if results:
        text = results[0][1].strip()
        return f"Forged — Edited. The '{text}' field appears altered."
    return "Forged — Edited. A field appears altered."


# ---------------------------------------------------------------------------
# Suspicious patch + reasoning
# ---------------------------------------------------------------------------

def _extract_suspicious_patch(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    # Converts the heatmap into a binary mask for the hottest region,
    # then extracts the most suspicious patch from the original RGB image.
    thresh = max(220, int(0.90 * heatmap.max()))
    mask = (heatmap >= thresh).astype(np.uint8) * 255

    # Clean the mask and slightly expand it so the extracted region
    # contains more visual context around the suspicious area.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no focused suspicious region is found, fall back to the center area of the image.
        h, w = rgb.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        patch = rgb[y1:y2, x1:x2]
        return cv2.resize(patch, (224, 224))

    # Use the largest suspicious contour and extract a padded bounding box around it.
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad = 10
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, rgb.shape[1])
    y2 = min(y + h + pad, rgb.shape[0])

    patch = rgb[y1:y2, x1:x2]
    if patch.size == 0:
        # Fallback again if the extracted patch is empty for any reason.
        h, w = rgb.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        patch = rgb[y1:y2, x1:x2]

    return cv2.resize(patch, (224, 224))


def _level(v: float) -> str:
    # Converts a numeric forensic score into an interpretable severity level.
    if v >= 0.55:
        return "High"
    if v >= 0.30:
        return "Moderate"
    return "Low"

def _stage1_explanation(real_prob: float, forged_prob: float) -> str:
    # Explains how confident stage 1 is in distinguishing real vs forged
    # based on the margin between the two probabilities.
    margin = abs(real_prob - forged_prob)

    if margin < 0.10:
        strength = "Weak"
    elif margin < 0.30:
        strength = "Moderate"
    else:
        strength = "Strong"

    if real_prob > forged_prob:
        decision = "The document more closely matches authentic patterns than forged ones."
    else:
        decision = "The document more closely matches forged patterns than authentic ones."

    return f"{strength} confidence in distinguishing real vs forged documents. \n- {decision}"


def _stage2_explanation(edited_prob: float, ai_prob: float) -> str:
    # Explains how confident stage 2 is in distinguishing edited vs AI-generated
    # based on the probability gap between the two classes.
    margin = abs(edited_prob - ai_prob)

    if margin < 0.10:
        strength = "Weak"
    elif margin < 0.30:
        strength = "Moderate"
    else:
        strength = "Strong"

    if edited_prob > ai_prob:
        decision = "The document is closer to edited patterns than AI-generated ones."
    else:
        decision = "The document is closer to AI-generated patterns than edited ones."

    return f"{strength} confidence in distinguishing edited vs AI-generated documents. \n- {decision}"

def _reasoning_text(
    verdict: str,
    real_prob: float,
    forged_prob: float,
    edited_prob: float | None,
    ai_prob: float | None,
    rgb: np.ndarray,
) -> str:
    # Recomputes the three forensic maps and derives simple aggregate scores
    # so the final prediction can be explained in a human-readable way.
    forensic = _forensic_stack(rgb)
    ela = forensic[0]
    srm = forensic[1]
    fft_map = forensic[2]

    compression_score = float(np.mean(ela))
    noise_score = float(np.mean(srm))
    frequency_score = float(np.mean(fft_map))

    lines = []
    lines.append("Forensic signals:")
    lines.append(f"- Compression inconsistency: {_level(compression_score)}")
    lines.append(f"- Noise anomaly: {_level(noise_score)}")
    lines.append(f"- Frequency anomaly: {_level(frequency_score)}")
    lines.append("")

    lines.append("")
    lines.append("Stage 1 interpretation:")
    lines.append(f"- {_stage1_explanation(real_prob, forged_prob)}")

    # Include stage 2 interpretation only if the document was treated as forged
    # and stage 2 probabilities are available.
    if edited_prob is not None and ai_prob is not None:
        lines.append("")
        lines.append("Stage 2 interpretation:")
        lines.append(f"- {_stage2_explanation(edited_prob, ai_prob)}")

    return "\n".join(lines)
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(ckpt: Path | None = None, device: str = "cpu") -> UnifiedForgeryModel:
    # Builds the model in evaluation mode and optionally loads trained weights
    # from the provided checkpoint file.
    model = UnifiedForgeryModel().to(device).eval()
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loading checkpoint: {ckpt}")
    return model


def predict_image(
    image_path: Path,
    model: UnifiedForgeryModel,
    device: str = "cpu",
    image_size: int = 224,
    stage1_threshold: float = 0.80,
    inconclusive_margin: float = 0.15,
) -> Dict[str, Any]:
    # Loads the image and prepares both model inputs:
    # - RGB tensor
    # - 3-channel forensic tensor generated on the fly
    rgb = _load_rgb(image_path, size=image_size)
    rgb_t = (
        torch.from_numpy(rgb.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    fore_t = torch.from_numpy(_forensic_stack(rgb)).unsqueeze(0).to(device)

    # Run the model once to obtain logits from both stages.
    with torch.no_grad():
        out = model(rgb_t, fore_t)

    # Stage 1: real vs forged
    p1 = F.softmax(out["stage1_logits"], dim=1)[0]
    real_prob, forged_prob = float(p1[0]), float(p1[1])
    print(f"Stage1 -> real={real_prob:.4f}, forged={forged_prob:.4f}")

    # If the two stage 1 probabilities are too close, treat the result as inconclusive
    # and still provide an explainability heatmap plus a suspicious patch.
    if abs(real_prob - forged_prob) < inconclusive_margin:
        heatmap = _gradcam_vit(
            model,
            rgb_t,
            fore_t,
            target_class=int(forged_prob >= real_prob),
            head="stage1",
        )
        return {
            "verdict": "Inconclusive",
            "confidence": max(real_prob, forged_prob),
            "caption": "The model is uncertain for this image.",
            "reasoning": _reasoning_text("Inconclusive", real_prob, forged_prob, None, None, rgb),
            "suspicious_patch": _extract_suspicious_patch(rgb, heatmap),
            "real_prob": real_prob,
            "forged_prob": forged_prob,
            "edited_prob": None,
            "ai_prob": None,
        }

    # If forged confidence does not cross the configured threshold,
    # label the image as original and stop before stage 2.
    if forged_prob < stage1_threshold:
        heatmap = _gradcam_vit(model, rgb_t, fore_t, target_class=0, head="stage1_final")
        return {
            "verdict": "Original Document",
            "confidence": real_prob,
            "caption": "No strong forgery evidence detected.",
            "reasoning": _reasoning_text("Original Document", real_prob, forged_prob, None, None, rgb),
            "suspicious_patch": _extract_suspicious_patch(rgb, heatmap),
            "real_prob": real_prob,
            "forged_prob": forged_prob,
            "edited_prob": None,
            "ai_prob": None,
        }

    # Stage 2 runs only if stage 1 classifies the image as forged.
    p2 = F.softmax(out["stage2_logits"], dim=1)[0]
    edited_prob, ai_prob = float(p2[0]), float(p2[1])
    print(f"Stage2 -> edited={edited_prob:.4f}, ai_generated={ai_prob:.4f}")

    if edited_prob >= ai_prob:
        # For edited forgeries, use OCR on the hottest suspicious region
        # to produce a more specific caption naming the altered field if possible.
        heatmap = _gradcam_vit(model, rgb_t, fore_t, target_class=0, head="stage2_final")
        caption = _ocr_caption(heatmap, rgb)
        verdict = "Forged — Edited"
        conf = edited_prob
    else:
        # For AI-generated-like forgeries, use a generic forensic caption.
        heatmap = _gradcam_vit(model, rgb_t, fore_t, target_class=1, head="stage2_final")
        caption = "AI-generated-like forensic pattern detected."
        verdict = "Forged — AI Generated"
        conf = ai_prob

    # Return the final prediction package used by the CLI and Gradio app.
    return {
        "verdict": verdict,
        "confidence": conf,
        "caption": caption,
        "reasoning": _reasoning_text(verdict, real_prob, forged_prob, edited_prob, ai_prob, rgb),
        "suspicious_patch": _extract_suspicious_patch(rgb, heatmap),
        "real_prob": real_prob,
        "forged_prob": forged_prob,
        "edited_prob": edited_prob,
        "ai_prob": ai_prob,
    }