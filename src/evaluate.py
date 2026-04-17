from __future__ import annotations

import random
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from config_utils import load_config, resolve_path
from predict import load_model, predict_image


def get_images(folder: Path) -> list[Path]:
    # Collects all files directly inside the given folder.
    # This is used to gather test images for each class.
    return [p for p in folder.glob("*") if p.is_file()]


def run_evaluation(sample_size: int = 30) -> None:
    # Load configuration values needed for evaluation:
    # dataset paths, checkpoint path, inference settings, and image size.
    cfg = load_config()
    paths = cfg.get("paths", {})
    checkpoints = cfg.get("checkpoints", {})
    inference = cfg.get("inference", {})
    training = cfg.get("training", {})

    # Resolve the root test folder and locate each class-specific directory.
    rgb_test_root = resolve_path(paths.get("rgb_test_root", "data/final_data/rgb/test"))
    real_dir = rgb_test_root / "real"
    edited_dir = rgb_test_root / "edited"
    ai_dir = rgb_test_root / "ai_generated"

    # Load the trained model checkpoint and inference thresholds used during testing.
    ckpt = resolve_path(checkpoints.get("stage2", "checkpoints/stage2.pt"))
    device = inference.get("device", "cpu")
    image_size = training.get("image_size", 224)
    stage1_threshold = inference.get("stage1_threshold", 0.80)
    inconclusive_margin = inference.get("inconclusive_margin", 0.15)

    model = load_model(ckpt, device=device)

    # Randomly sample a fixed number of images from each class,
    # or all available images if the folder has fewer than sample_size.
    real_imgs = random.sample(get_images(real_dir), min(sample_size, len(get_images(real_dir))))
    edited_imgs = random.sample(get_images(edited_dir), min(sample_size, len(get_images(edited_dir))))
    ai_imgs = random.sample(get_images(ai_dir), min(sample_size, len(get_images(ai_dir))))

    # These lists store ground-truth labels and predicted labels
    # for the three evaluation views:
    # 1. Stage 1 binary classification
    # 2. Stage 2 binary classification
    # 3. Final unified 3-class classification
    y_true_stage1, y_pred_stage1 = [], []
    y_true_stage2, y_pred_stage2 = [], []
    y_true_final, y_pred_final = [], []

    # Evaluate each sampled image class by class.
    for true_label, images in [
        ("real", real_imgs),
        ("edited", edited_imgs),
        ("ai", ai_imgs),
    ]:
        print(f"[INFO] Evaluating {true_label.upper()}...")

        for img in images:
            # Run the full prediction pipeline on the image.
            out = predict_image(
                img,
                model,
                device=device,
                image_size=image_size,
                stage1_threshold=stage1_threshold,
                inconclusive_margin=inconclusive_margin,
            )

            verdict = out["verdict"].lower()

            # -------- Stage 1: real vs forged --------
            # Ground truth is reduced to binary:
            # real stays real, while edited and ai are grouped as forged.
            if true_label == "real":
                y_true_stage1.append("real")
            else:
                y_true_stage1.append("forged")

            # Convert the model's textual verdict into a binary stage 1 prediction.
            if "real" in verdict or "original document" in verdict:
                y_pred_stage1.append("real")
            else:
                y_pred_stage1.append("forged")

            # -------- Stage 2: edited vs ai --------
            # Stage 2 is evaluated only on forged samples,
            # so real images are excluded from these lists.
            if true_label in ["edited", "ai"]:
                y_true_stage2.append(true_label)

                # Extract edited vs ai from the verdict string.
                # If neither keyword appears, default to edited as fallback.
                if "ai" in verdict:
                    y_pred_stage2.append("ai")
                elif "edited" in verdict:
                    y_pred_stage2.append("edited")
                else:
                    y_pred_stage2.append("edited")  # fallback

            # -------- Final unified model --------
            # Keep the original 3-class ground truth for overall evaluation.
            y_true_final.append(true_label)

            # Map the textual verdict into one of the three final class labels.
            # If no class keyword is found, default to real as fallback.
            if "real" in verdict or "original document" in verdict:
                y_pred_final.append("real")
            elif "ai" in verdict:
                y_pred_final.append("ai")
            elif "edited" in verdict:
                y_pred_final.append("edited")
            else:
                y_pred_final.append("real")  # fallback

    # Print binary evaluation for stage 1.
    print("\n" + "=" * 60)
    print("STAGE 1: REAL vs FORGED")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_true_stage1, y_pred_stage1))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_stage1, y_pred_stage1, labels=["real", "forged"]))
    print("Classification Report:")
    print(classification_report(y_true_stage1, y_pred_stage1, labels=["real", "forged"]))

    # Print binary evaluation for stage 2.
    print("\n" + "=" * 60)
    print("STAGE 2: EDITED vs AI")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_true_stage2, y_pred_stage2))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_stage2, y_pred_stage2, labels=["edited", "ai"]))
    print("Classification Report:")
    print(classification_report(y_true_stage2, y_pred_stage2, labels=["edited", "ai"]))

    # Print final 3-class evaluation for the complete pipeline.
    print("\n" + "=" * 60)
    print("UNIFIED MODEL: REAL vs EDITED vs AI")
    print("=" * 60)
    print("Accuracy:", accuracy_score(y_true_final, y_pred_final))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_final, y_pred_final, labels=["real", "edited", "ai"]))
    print("Classification Report:")
    print(classification_report(y_true_final, y_pred_final, labels=["real", "edited", "ai"]))


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()