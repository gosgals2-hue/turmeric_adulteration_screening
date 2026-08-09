"""
extract_features.py — Masked color/texture features for classical ML baselines.

Extracts 11 features PER IMAGE from the segmented sample region only, so the
white/colored background never contaminates the color statistics. The mask
comes from segment_bg.find_sample_roi (illumination-independent), replacing the
old whole-image HSV approach that averaged in background pixels.

Run from project root:
    python src/extract_features.py                       # default: data/split/train
    python src/extract_features.py --src data/split/val  --out features/val.csv
    python src/extract_features.py --src data/pure --label 0 --out features/pure.csv

Label inference (when --label is not given): a file is class 1 (adulterated)
if its path contains "adulterated" or "starch", 0 (pure) if it contains "pure".
Unlabelable files are skipped with a warning.

The 11 features (all computed over masked sample pixels only):
    mean_hue, mean_blue, mean_green, mean_red,
    mean_saturation, mean_value,
    hue_std, saturation_std, value_std,
    brightness (mean gray), contrast (std gray)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

FEATURE_COLUMNS = [
    "mean_hue", "mean_blue", "mean_green", "mean_red",
    "mean_saturation", "mean_value",
    "hue_std", "saturation_std", "value_std",
    "brightness", "contrast",
]


def extract_features(image_path):
    """
    Compute the 11 features over the segmented sample region of one image.

    Returns a list of 11 floats, or None if the image can't be read or the
    segmentation yields no usable pixels.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  WARN: could not load {image_path}")
        return None

    # Segment the sample. ok=False means find_sample_roi fell back to a center
    # crop; the mask is still a reasonable region, so we proceed but flag it.
    mask, _bbox, ok = find_sample_roi(img)
    valid = mask > 0
    if valid.sum() < 100:  # essentially nothing segmented
        print(f"  WARN: empty mask for {Path(image_path).name} — skipping")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    return [
        float(np.mean(h[valid])),
        float(np.mean(b[valid])),
        float(np.mean(g[valid])),
        float(np.mean(r[valid])),
        float(np.mean(s[valid])),
        float(np.mean(v[valid])),
        float(np.std(h[valid])),
        float(np.std(s[valid])),
        float(np.std(v[valid])),
        float(np.mean(gray[valid])),
        float(np.std(gray[valid])),
    ], ok


def infer_label(path: Path):
    """Return 0 (pure), 1 (adulterated), or None if undecidable."""
    p = str(path).lower()
    if "pure" in p:
        return 0
    if "adulterated" in p or "starch" in p:
        return 1
    return None


def process_folder(folder: Path, forced_label):
    rows = []
    fallbacks = 0
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        print(f"  WARN: no images found under {folder}")
        return rows, fallbacks

    for i, path in enumerate(files, 1):
        label = forced_label if forced_label is not None else infer_label(path)
        if label is None:
            print(f"  WARN: can't infer label for {path} — skipping")
            continue
        result = extract_features(path)
        if result is None:
            continue
        feats, ok = result
        fallbacks += (not ok)
        rows.append(feats + [label, str(path.relative_to(folder)), int(ok)])
        if i % 50 == 0:
            print(f"    processed {i}/{len(files)}")

    return rows, fallbacks


def main():
    ap = argparse.ArgumentParser(
        description="Extract masked color/texture features for ML baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", default="data/split/train",
                    help="Folder of images to process (default: data/split/train)")
    ap.add_argument("--out", default="features/train.csv",
                    help="Output CSV path (default: features/train.csv)")
    ap.add_argument("--label", type=int, choices=[0, 1], default=None,
                    help="Force a single label for every file (else inferred from path)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"ERROR: source folder '{src}' does not exist.")

    print(f"Extracting masked features")
    print(f"  Source : {src.resolve()}")
    print(f"  Output : {Path(args.out).resolve()}")
    print(f"  Label  : {'inferred from path' if args.label is None else args.label}")

    rows, fallbacks = process_folder(src, args.label)
    if not rows:
        sys.exit("ERROR: no features extracted.")

    columns = FEATURE_COLUMNS + ["label", "rel_path", "roi_ok"]
    df = pd.DataFrame(rows, columns=columns)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df)} rows written to {out_path}")
    print(f"  Class balance: {df['label'].value_counts().to_dict()}")
    if fallbacks:
        print(f"  NOTE: {fallbacks} image(s) used the segmentation fallback "
              f"(roi_ok=0) — inspect if that's a large fraction.")


if __name__ == "__main__":
    main()
