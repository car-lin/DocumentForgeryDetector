from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import load_config, resolve_path
import dataprep
import train as train_module
from predict import load_model, predict_image
import app as app_module
import evaluate as evaluate_module


def run_prepare(clean: bool) -> None:
    # Reuses dataprep.main() by temporarily replacing sys.argv,
    # so the prepare step can be triggered programmatically
    # exactly as if the dataprep script was run from the command line.
    import sys
    argv = ["dataprep.py"]
    if clean:
        argv.append("--clean")
    old = sys.argv
    sys.argv = argv
    try:
        dataprep.main()
    finally:
        # Restore the original command-line arguments after execution
        # so other parts of the program are not affected.
        sys.argv = old


def run_train(stage: str) -> None:
    # Loads all training-related paths and hyperparameters from config.yaml,
    # then dispatches training for stage 1, stage 2, or both.
    cfg = load_config()
    paths = cfg.get("paths", {})
    training = cfg.get("training", {})
    rgb_root = resolve_path(paths.get("rgb_train_root", "data/final_data/rgb/train"))
    ela_root = resolve_path(paths.get("ela_train_root", "data/final_data/ela/train"))
    srm_root = resolve_path(paths.get("srm_train_root", "data/final_data/srm/train"))
    fft_root = resolve_path(paths.get("fft_train_root", "data/final_data/fft/train"))
    device = training.get("device", "cpu")
    image_size = training.get("image_size", 224)
    batch_size = training.get("batch_size", 16)
    num_workers = training.get("num_workers", 4)
    lr = training.get("lr", 1e-4)
    weight_decay = training.get("weight_decay", 1e-2)
    stage1_epochs = training.get("stage1_epochs", 3)
    stage2_epochs = training.get("stage2_epochs", 3)
    ckpts = cfg.get("checkpoints", {})
    stage1_ckpt = resolve_path(ckpts.get("stage1", "checkpoints/stage1.pt"))
    stage2_ckpt = resolve_path(ckpts.get("stage2", "checkpoints/stage2.pt"))

    # Small temporary container class used to create an object
    # with attributes expected by the existing training functions.
    class Args: pass

    if stage in {"stage1", "all"}:
        # Prepare arguments for stage 1 training:
        # binary classification -> real vs forged.
        a = Args()
        a.rgb_root=rgb_root; a.ela_root=ela_root; a.srm_root=srm_root; a.fft_root=fft_root
        a.device=device; a.image_size=image_size; a.batch_size=batch_size; a.num_workers=num_workers
        a.lr=lr; a.weight_decay=weight_decay; a.epochs=stage1_epochs; a.save_path=stage1_ckpt
        train_module.train_stage1(a)

    if stage in {"stage2", "all"}:
        # Prepare arguments for stage 2 training:
        # multiclass classification over forged images, using the stage 1 checkpoint
        # as part of the two-stage pipeline.
        a = Args()
        a.rgb_root=rgb_root; a.ela_root=ela_root; a.srm_root=srm_root; a.fft_root=fft_root
        a.device=device; a.image_size=image_size; a.batch_size=batch_size; a.num_workers=num_workers
        a.lr=lr; a.weight_decay=weight_decay; a.epochs=stage2_epochs; a.save_path=stage2_ckpt
        a.stage1_ckpt=stage1_ckpt; a.skip_stage1_ckpt=False
        train_module.train_stage2(a)


def run_infer(image: str) -> None:
    # Loads the trained stage 2 model and runs inference on one image.
    # It prints the final verdict, a short caption, and the detailed reasoning.
    cfg = load_config()
    ckpt = resolve_path(cfg.get("checkpoints", {}).get("stage2", "checkpoints/stage2.pt"))
    inf = cfg.get("inference", {})
    device = inf.get("device", "cpu")
    image_size = cfg.get("training", {}).get("image_size", 224)
    model = load_model(ckpt, device=device)
    out = predict_image(Path(image), model, device=device, image_size=image_size,
                        stage1_threshold=inf.get("stage1_threshold", 0.80),
                        inconclusive_margin=inf.get("inconclusive_margin", 0.15))
    print(out["verdict"])
    print(out["caption"])
    print(out["reasoning"])


def main():
    # Builds a command-line interface with subcommands so the whole project
    # can be run from one entry point: prepare, train, infer, deploy, evaluate, or all.
    parser = argparse.ArgumentParser(description="Document forgery pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--clean", action="store_true")

    p_train = sub.add_parser("train")
    p_train.add_argument("--stage", choices=["stage1","stage2","all"], default="all")

    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--image", required=True)

    sub.add_parser("deploy")

    p_evaluate = sub.add_parser("evaluate")
    p_evaluate.add_argument("--sample-size", type=int, default=30)

    p_all = sub.add_parser("all")
    p_all.add_argument("--clean", action="store_true")
    p_all.add_argument("--infer-image")

    args = parser.parse_args()

    # Route execution based on the selected subcommand.
    if args.command == "prepare":
        run_prepare(clean=args.clean)
    elif args.command == "train":
        run_train(stage=args.stage)
    elif args.command == "infer":
        run_infer(args.image)
    elif args.command == "deploy":
        # Launches the Gradio application for interactive use.
        app_module.main()
    elif args.command == "evaluate":
        # Runs evaluation on a sample of dataset images.
        evaluate_module.run_evaluation(sample_size=args.sample_size)
    elif args.command == "all":
        # Executes the full pipeline end-to-end:
        # dataset preparation -> training -> optional inference on one image.
        run_prepare(clean=args.clean)
        run_train(stage="all")
        if args.infer_image:
            run_infer(args.infer_image)


if __name__ == "__main__":
    main()