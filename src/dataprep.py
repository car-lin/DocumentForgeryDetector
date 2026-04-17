from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from config_utils import load_config, resolve_path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RTM_REAL_LABELS = {"good"}
RTM_EDITED_LABELS = {"splice", "cover", "edit", "cpmv", "insert", "inpaint"}


def is_image(path: Path) -> bool:
    # Checks whether the given path is a valid image file
    # by verifying both file existence and allowed extension.
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def collect_images(root: Path) -> list[Path]:
    # Recursively collects all valid image files under a folder.
    # If the folder is missing, it warns and returns an empty list.
    if not root.exists():
        print(f"[WARN] Missing folder: {root}")
        return []
    return [p for p in root.rglob("*") if is_image(p)]


def ensure_dir(path: Path, clean: bool = False) -> None:
    # Makes sure the target directory exists.
    # If clean=True and the folder already exists, it is deleted first.
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_sampled(src_paths: list[Path], out_dir: Path, prefix: str, max_count: int | None, seed: int) -> int:
    # Copies images into the output folder, optionally sampling a fixed number
    # using a reproducible random seed. Files are renamed with a prefix and index
    # so the final dataset has consistent file naming.
    if not src_paths:
        return 0
    rng = random.Random(seed)
    chosen = list(src_paths)
    if max_count is not None and len(chosen) > max_count:
        chosen = rng.sample(chosen, max_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, src in enumerate(sorted(chosen)):
        dst = out_dir / f"{prefix}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def collect_rtm_by_prefix(rtm_root: Path, allowed_labels: set[str]) -> list[Path]:
    # Collects RTM images based on the label encoded in the filename prefix.
    # Example: if filename starts with "good_", it is treated as real;
    # if it starts with "splice_", "edit_", etc., it is treated as edited.
    img_root = rtm_root / "JPEGImages"
    if not img_root.exists():
        raise FileNotFoundError(f"RTM JPEGImages folder not found: {img_root}")
    found = []
    for p in img_root.iterdir():
        if not is_image(p):
            continue
        prefix = p.stem.lower().split("_")[0]
        if prefix in allowed_labels:
            found.append(p)
    return found


def collect_sroie_images(sroie_root: Path) -> list[Path]:
    # Collects SROIE receipt images from both train and test folders,
    # treating them as real/authentic document images.
    found = []
    for split in ["train", "test"]:
        found.extend(collect_images(sroie_root / split / "img"))
    return found


def generate_ai_forged_with_script(script_path: Path, count: int, seed: int, steps: int = 20) -> None:
    # Runs a separate local script that generates AI-forged images.
    # The command is built dynamically and executed as a subprocess so that
    # AI-generated forged samples can be added into the dataset pipeline.
    if not script_path.exists():
        raise FileNotFoundError(f"GenAI forge script not found: {script_path}")
    cmd = ["python", str(script_path), "--count", str(count), "--steps", str(steps), "--seed", str(seed)]
    print("[INFO] Running GenAI forge script...")
    print("[DEBUG] Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)




def split_train_test(rgb_root: Path, train_ratio: float = 0.8, seed: int = 42) -> None:
    # Splits each RGB class folder into train and test subsets using the given ratio.
    # After copying files into rgb/train/... and rgb/test/..., the original flat
    # class folders are removed to leave only the train/test structure.
    print("[INFO] Splitting RGB dataset into train/test...")
    rng = random.Random(seed)

    for cls in ["real", "edited", "ai_generated"]:
        src_dir = rgb_root / cls
        if not src_dir.exists():
            print(f"[WARN] Missing class folder for split: {src_dir}")
            continue

        files = [p for p in src_dir.glob("*") if p.is_file()]
        rng.shuffle(files)

        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        train_dir = rgb_root / "train" / cls
        test_dir = rgb_root / "test" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for f in train_files:
            shutil.copy2(f, train_dir / f.name)
        for f in test_files:
            shutil.copy2(f, test_dir / f.name)

        print(f"[INFO] {cls}: train={len(train_files)}, test={len(test_files)}")

    # Remove the original unsplit class folders once train/test folders are ready.
    for cls in ["real", "edited", "ai_generated"]:
        src_dir = rgb_root / cls
        if src_dir.exists():
            shutil.rmtree(src_dir)


def _iter_rgb_images(rgb_root: Path):
    # Internal helper generator that yields all image files
    # from both rgb/train and rgb/test folders.
    for split in ["train", "test"]:
        split_dir = rgb_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if is_image(p):
                yield p


def generate_ela(rgb_root: Path, ela_root: Path, quality: int = 90) -> None:
    # Generates ELA (Error Level Analysis) images.
    # Each RGB image is recompressed as JPEG, then the difference between
    # the original and recompressed image is amplified and saved as grayscale.
    ensure_dir(ela_root)
    for src in _iter_rgb_images(rgb_root):
        rel = src.relative_to(rgb_root).with_suffix(".png")
        dst = ela_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            img.save(tmp_path, "JPEG", quality=quality)
            recompressed = Image.open(tmp_path).convert("RGB")
            diff = ImageChops.difference(img, recompressed)
            extrema = diff.getextrema()
            max_diff = max(ex[1] for ex in extrema) or 1
            ela_img = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff).convert("L")
            ela_img.save(dst)
        finally:
            tmp_path.unlink(missing_ok=True)


def generate_srm(rgb_root: Path, srm_root: Path) -> None:
    # Generates SRM-like forensic maps using a handcrafted high-pass residual kernel.
    # This emphasizes subtle noise and manipulation artifacts that may not be visible
    # in the original image.
    ensure_dir(srm_root)
    kernel = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        dtype=np.float32,
    )
    for src in _iter_rgb_images(rgb_root):
        rel = src.relative_to(rgb_root).with_suffix(".png")
        dst = srm_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        srm = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        srm = cv2.normalize(np.abs(srm), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(dst), srm)


def generate_fft(rgb_root: Path, fft_root: Path) -> None:
    # Generates FFT magnitude images from grayscale document images.
    # This captures frequency-domain patterns that can help expose unnatural
    # generation or tampering artifacts.
    ensure_dir(fft_root)
    for src in _iter_rgb_images(rgb_root):
        rel = src.relative_to(rgb_root).with_suffix(".png")
        dst = fft_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag = 20 * np.log(np.abs(fft) + 1.0)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(dst), mag)


def generate_features(rgb_root: Path, ela_root: Path, srm_root: Path, fft_root: Path, ela_quality: int) -> None:
    # Runs all forensic feature generation steps in sequence
    # for the prepared RGB dataset.
    print("[INFO] Generating ELA...")
    generate_ela(rgb_root, ela_root, quality=ela_quality)
    print("[INFO] Generating SRM...")
    generate_srm(rgb_root, srm_root)
    print("[INFO] Generating FFT...")
    generate_fft(rgb_root, fft_root)


def summarize_counts(base: Path) -> None:
    # Prints the number of files available in each modality/class combination
    # so the final dataset composition can be verified.
    for modality in ["rgb", "ela", "srm", "fft"]:
        for cls in ["real", "edited", "ai_generated"]:
            folder = base / modality / cls
            count = len([p for p in folder.rglob("*") if p.is_file()]) if folder.exists() else 0
            print(f"{modality:>3} / {cls:<12}: {count}")


def main() -> None:
    # Load dataset paths and defaults from config.yaml.
    cfg = load_config()
    paths = cfg.get("paths", {})

    # Define CLI arguments so dataset sources, output location, counts,
    # and preprocessing settings can be controlled from the command line.
    p = argparse.ArgumentParser(description="Build final dataset for document forgery pipeline")
    p.add_argument("--rtm-root", type=Path, default=resolve_path(paths.get("rtm_root")))
    p.add_argument("--sroie-root", type=Path, default=resolve_path(paths.get("sroie_root")))
    p.add_argument("--output-root", type=Path, default=resolve_path(paths.get("output_root", "data/final_data")))
    p.add_argument("--genai-script", type=Path, default=resolve_path(paths.get("genai_script", "src/genAI_forge_class.py")))
    p.add_argument("--rtm-real-count", type=int, default=3000)
    p.add_argument("--rtm-edited-count", type=int, default=6000)
    p.add_argument("--sroie-count", type=int, default=1000)
    p.add_argument("--ai-count", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ela-quality", type=int, default=90)
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    # Stop execution if required source dataset paths are missing.
    if args.rtm_root is None or args.sroie_root is None:
        raise SystemExit("Please set paths.rtm_root and paths.sroie_root in config.yaml or pass them via CLI.")

    # Define output folder structure for all modalities.
    out_root = args.output_root
    rgb_root = out_root / "rgb"
    ela_root = out_root / "ela"
    srm_root = out_root / "srm"
    fft_root = out_root / "fft"

    # Optionally wipe the entire output folder before rebuilding.
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    # Create top-level output directories.
    for folder in [rgb_root, ela_root, srm_root, fft_root]:
        ensure_dir(folder)

    # Create RGB class folders where source images are first collected.
    rgb_real = rgb_root / "real"
    rgb_edited = rgb_root / "edited"
    rgb_ai = rgb_root / "ai_generated"
    ensure_dir(rgb_real)
    ensure_dir(rgb_edited)
    ensure_dir(rgb_ai)

    # Step 1: Build real and edited classes from RTM using filename labels.
    print("[1/5] Collecting RTM real and edited by filename prefix...")
    rtm_real_imgs = collect_rtm_by_prefix(args.rtm_root, RTM_REAL_LABELS)
    rtm_edited_imgs = collect_rtm_by_prefix(args.rtm_root, RTM_EDITED_LABELS)
    n_rtm_real = copy_sampled(rtm_real_imgs, rgb_real, prefix="rtm_real", max_count=args.rtm_real_count, seed=args.seed)
    n_rtm_edited = copy_sampled(rtm_edited_imgs, rgb_edited, prefix="rtm_edit", max_count=args.rtm_edited_count, seed=args.seed + 1)
    print(f"[INFO] RTM real added: {n_rtm_real}")
    print(f"[INFO] RTM edited added: {n_rtm_edited}")

    # Step 2: Add extra real samples from SROIE receipt dataset.
    print("[2/5] Collecting SROIE real images from train/img and test/img...")
    sroie_imgs = collect_sroie_images(args.sroie_root)
    n_sroie = copy_sampled(sroie_imgs, rgb_real, prefix="sroie", max_count=args.sroie_count, seed=args.seed + 2)
    print(f"[INFO] SROIE real added: {n_sroie}")

    # Step 3: Generate AI-forged samples using the external local script.
    # These are expected to be saved into rgb/ai_generated.
    print("[3/5] Generating AI-forged images from rgb/real using local GenAI script...")
    generate_ai_forged_with_script(script_path=args.genai_script, count=args.ai_count, seed=args.seed + 3, steps=20)
    n_ai = len(collect_images(rgb_ai))
    print(f"[INFO] AI-forged images created: {n_ai}")

    # Step 3.5: Split the RGB dataset into train/test folders.
    print("[3.5/5] Creating train/test split (80/20)...")
    split_train_test(rgb_root, train_ratio=0.8, seed=args.seed)

    # Step 4: Generate forensic feature modalities from the split RGB dataset.
    print("[4/5] Generating ELA / SRM / FFT...")
    generate_features(rgb_root=rgb_root, ela_root=ela_root, srm_root=srm_root, fft_root=fft_root, ela_quality=args.ela_quality)

    # Final dataset summary.
    print("\nFinal counts:")
    summarize_counts(out_root)
    print("\n[5/5] Done.")
    print(f"Dataset ready at: {out_root}")


if __name__ == "__main__":
    main()