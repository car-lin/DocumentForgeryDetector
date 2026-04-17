from pathlib import Path
import argparse
import random

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

from config_utils import load_config, resolve_path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path):
    # Checks whether the file extension matches a supported image type.
    return path.suffix.lower() in IMG_EXTS


def get_images(folder):
    # Recursively collects all valid image files from the input folder.
    return [p for p in folder.rglob("*") if p.is_file() and is_image(p)]


def read_rgb(path):
    # Reads an image with OpenCV and converts it from BGR to RGB
    # so later processing uses standard RGB ordering.
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    # Saves an RGB image to disk by converting it back to OpenCV's
    # expected BGR format and creating parent folders if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def get_device():
    # Selects the best available device for inference:
    # CUDA first, then Apple MPS, otherwise CPU.
    # Also chooses a suitable tensor precision for that device.
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model(model_id, device, dtype):
    # Loads the diffusion inpainting pipeline and moves it to the chosen device.
    # For MPS, attention slicing is enabled when available to reduce memory usage.
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "mps":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe


def find_text_boxes(rgb):
    # Detects text-like rectangular regions in the document:
    # 1. convert to grayscale
    # 2. use adaptive thresholding to highlight dark text
    # 3. merge nearby text characters/words using morphological closing
    # 4. extract bounding boxes and keep only boxes that look like plausible text fields
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        # Reject boxes that are too small to represent useful text content.
        if bw < 60 or bh < 12:
            continue

        # Reject boxes that are too large, since those are likely full-page regions
        # or layout structures rather than a specific editable text field.
        if bw > int(0.9 * w) or bh > int(0.15 * h):
            continue
        boxes.append((x, y, bw, bh))

    # Sort boxes roughly in reading order: top to bottom, then left to right.
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def pick_box(boxes):
    # Chooses one candidate text box to edit.
    # It first prefers larger boxes, then randomly selects from the top few
    # so the output is varied across different runs.
    if not boxes:
        return None
    boxes = sorted(boxes, key=lambda b: b[2], reverse=True)
    return random.choice(boxes[: min(10, len(boxes))])


def make_crop(rgb, box):
    # Creates a padded crop around the selected text box.
    # Extra surrounding context is included so the inpainting model can
    # generate an edit that blends naturally with the document layout.
    x, y, w, h = box
    H, W = rgb.shape[:2]
    pad_x = max(20, w)
    pad_y = max(15, h * 2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    crop = rgb[y1:y2, x1:x2].copy()

    # inner_box stores the original text box coordinates relative to the cropped region.
    inner_box = (x - x1, y - y1, w, h)
    return crop, (x1, y1), inner_box


def make_mask(crop_size, inner_box):
    # Builds a binary mask covering the chosen text field inside the crop.
    # A small margin is added around the box so the edited region looks more natural.
    crop_w, crop_h = crop_size
    x, y, w, h = inner_box
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    x1 = max(0, x - 5)
    y1 = max(0, y - 3)
    x2 = min(crop_w, x + w + 5)
    y2 = min(crop_h, y + h + 3)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return Image.fromarray(mask)


def edit_crop(pipe, crop_rgb, inner_box, device, steps=20, strength=0.85):
    # Runs diffusion-based inpainting on the cropped document region:
    # - resize crop to model input size
    # - scale the target box accordingly
    # - create a mask over the field to be replaced
    # - prompt the model to generate a realistic printed value
    # - resize the edited crop back to original crop dimensions
    crop_h, crop_w = crop_rgb.shape[:2]
    model_size = 512
    crop_img = Image.fromarray(crop_rgb).convert("RGB")
    crop_small = crop_img.resize((model_size, model_size), Image.Resampling.LANCZOS)

    x, y, w, h = inner_box
    sx = model_size / crop_w
    sy = model_size / crop_h
    scaled_box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

    mask = make_mask((model_size, model_size), scaled_box)

    # Prompt tries to keep the edit document-like and visually consistent.
    prompt = "Replace the masked field with a realistic printed receipt or invoice value. Keep the document style similar."
    negative_prompt = "handwriting, artistic style, distorted page, random symbols, extra objects, blur"

    # A random generator seed makes each edited value slightly different.
    generator = torch.Generator(device=device).manual_seed(random.randint(0, 1000000))

    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=crop_small, mask_image=mask,
                  num_inference_steps=steps, guidance_scale=7.5, strength=strength, generator=generator).images[0]

    result = result.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    edited_crop = np.array(result.convert("RGB"))
    return edited_crop, np.array(mask.convert("RGB"))


def forge_one_image(pipe, img_path, out_path, debug_dir, device, steps, strength):
    # Applies the full forgery pipeline to one image:
    # 1. read document
    # 2. detect candidate text fields
    # 3. choose one field
    # 4. crop and inpaint that region
    # 5. paste the edited crop back into the full image
    # 6. optionally save debug visuals
    rgb = read_rgb(img_path)
    boxes = find_text_boxes(rgb)
    box = pick_box(boxes)
    if box is None:
        print(f"[WARN] No text box found for {img_path.name}")
        return False

    crop, (x1, y1), inner_box = make_crop(rgb, box)
    edited_crop, mask_rgb = edit_crop(pipe, crop, inner_box, device, steps=steps, strength=strength)

    result = rgb.copy()
    ch, cw = edited_crop.shape[:2]
    result[y1:y1 + ch, x1:x1 + cw] = edited_crop
    save_rgb(out_path, result)

    # Debug outputs help inspect what region was selected and how it was edited.
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        save_rgb(debug_dir / f"{stem}_before.png", crop)
        save_rgb(debug_dir / f"{stem}_after.png", edited_crop)
        save_rgb(debug_dir / f"{stem}_mask.png", mask_rgb)
        vis = rgb.copy()
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        save_rgb(debug_dir / f"{stem}_box.png", vis)
    return True


def main():
    # Loads configuration and allows key parameters to be overridden from CLI:
    # input/output paths, number of images, diffusion steps, edit strength, seed, and model choice.
    cfg = load_config()
    paths = cfg.get("paths", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=resolve_path(paths.get("rgb_real")))
    parser.add_argument("--output-dir", type=Path, default=resolve_path(paths.get("rgb_ai_generated")))
    parser.add_argument("--debug-dir", type=Path, default=resolve_path(paths.get("genai_debug_dir")))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-inpainting")
    args = parser.parse_args()

    # Input and output folders are required either from config or CLI.
    if args.input_dir is None or args.output_dir is None:
        raise SystemExit("Please set paths.rgb_real and paths.rgb_ai_generated in config.yaml or pass them via CLI.")

    random.seed(args.seed)

    # Load input images from the real document folder.
    images = get_images(args.input_dir)
    if not images:
        print(f"[ERROR] No images found in {args.input_dir}")
        return

    # Limit the number of processed images if requested.
    if args.count is not None:
        images = images[: args.count]

    # Initialize the inpainting model on the best available device.
    device, dtype = get_device()
    print(f"[INFO] Using {device}")
    print(f"[INFO] Loading model: {args.model_id}")
    pipe = load_model(args.model_id, device, dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = 0

    # Process each image one by one and save the forged result.
    for i, img_path in enumerate(images):
        out_path = args.output_dir / f"ai_forged_{i:06d}.png"
        try:
            ok = forge_one_image(pipe, img_path, out_path, args.debug_dir, device, args.steps, args.strength)
            if ok:
                done += 1
                print(f"[OK] {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"[WARN] Failed on {img_path.name}: {e}")

    print(f"\nDone. Created {done} forged images in {args.output_dir}")


if __name__ == "__main__":
    main()