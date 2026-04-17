from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from config_utils import load_config, resolve_path, project_root
from predict import load_model, predict_image


def main():
    # Read all runtime settings such as checkpoint path, device, image size,
    # and decision thresholds from the config file.
    cfg = load_config()

    # Get the stage2 checkpoint path from config.
    # If not provided, fall back to checkpoints/stage2.pt.
    ckpt = resolve_path(cfg.get("checkpoints", {}).get("stage2", "checkpoints/stage2.pt"))

    # Select inference device, usually cpu / cuda / mps.
    device = cfg.get("inference", {}).get("device", "cpu")

    # Image size used during preprocessing so inference matches training setup.
    image_size = cfg.get("training", {}).get("image_size", 224)

    # Threshold used in Stage 1 to decide whether the image is real or forged.
    stage1_threshold = cfg.get("inference", {}).get("stage1_threshold", 0.80)

    # Margin used to mark borderline predictions as inconclusive.
    inconclusive_margin = cfg.get("inference", {}).get("inconclusive_margin", 0.15)

    # Load the trained model if the checkpoint exists.
    # If not, load_model handles the fallback behavior.
    model = load_model(ckpt if ckpt and ckpt.exists() else None, device=device)

    # Temporary file used because predict_image expects an image path as input.
    tmp = project_root() / "tmp_input.png"

    def run(image: np.ndarray):
        # If the user has not uploaded an image, return placeholder outputs
        # matching all UI components.
        if image is None:
            return "No image", 0.0, "Upload an image first.", "", None, None, None, None, None

        # Save the uploaded numpy image temporarily so it can be passed
        # through the existing prediction pipeline.
        Image.fromarray(image).save(tmp)

        # Run the full inference pipeline:
        # Stage 1 -> real vs forged
        # Stage 2 -> if forged, edited vs AI-generated
        out = predict_image(tmp, model, device=device, image_size=image_size, stage1_threshold=stage1_threshold, inconclusive_margin=inconclusive_margin)

        # Remove the temporary file after inference completes.
        tmp.unlink(missing_ok=True)

        # Return all values in the exact order expected by the Gradio outputs.
        return (
            out["verdict"],
            round(out["confidence"], 4),
            out["caption"],
            out["reasoning"],
            out["suspicious_patch"],
            None if out["real_prob"] is None else round(out["real_prob"], 4),
            None if out["forged_prob"] is None else round(out["forged_prob"], 4),
            None if out["edited_prob"] is None else round(out["edited_prob"], 4),
            None if out["ai_prob"] is None else round(out["ai_prob"], 4),
        )

    with gr.Blocks(title="Document Forgery Detector") as demo:
        gr.Markdown("## Document Forgery Detector\nUpload any document image — photo, scan, or screenshot.")
        with gr.Row():
            img_in = gr.Image(type="numpy", label="Input image")
            suspicious_patch = gr.Image(type="numpy", label="Suspicious patch")
        verdict = gr.Textbox(label="Verdict")
        confidence = gr.Number(label="Confidence")
        caption = gr.Textbox(label="Forensic caption")
        with gr.Row():
            real_prob = gr.Number(label="Stage1 Real Prob")
            forged_prob = gr.Number(label="Stage1 Forged Prob")
            edited_prob = gr.Number(label="Stage2 Edited Prob")
            ai_prob = gr.Number(label="Stage2 AI Prob")
        reasoning = gr.Textbox(label="Reasoning", lines=10)

        # When the Analyse button is clicked, the uploaded image is passed to run(),
        # and the returned values are mapped to these output components in order.
        gr.Button("Analyse").click(run, inputs=[img_in], outputs=[verdict, confidence, caption, reasoning, suspicious_patch, real_prob, forged_prob, edited_prob, ai_prob])

    # Start the Gradio app locally.
    demo.launch()


if __name__ == "__main__":
    main()