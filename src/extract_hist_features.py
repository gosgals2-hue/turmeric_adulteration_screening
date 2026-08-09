"""
extract_hist_features.py — Masked 32-bin per-channel RGB histogram features (Tier-2).

Reconstructs the 96-dimensional colour-histogram feature table used by
train_tier2_svm.py (eda/hist_features.csv). That CSV existed without a source
script in the repo; this script regenerates it deterministically and, crucially,
exposes the histogram function so the degradation pipeline (degrade_eval.py) can
re-featurize DEGRADED images with the identical representation.

Representation (per image):
  1. Segment the sample with segment_bg.find_sample_roi (illumination-independent).
  2. For each channel R, G, B: a 32-bin histogram over [0, 256) computed over the
     MASKED sample pixels only, L1-normalized so each channel's 32 bins sum to 1.
  3. Concatenate -> 96 features: hR0..hR31, hG0..hG31, hB0..hB31.

The masked-and-normalized design means background pixels never enter the histogram
and overall ROI size doesn't matter — only the colour *distribution* does, which is
exactly the Tier-2 hypothesis (distribution >> means).

Run from project root:
    python src/extract_hist_features.py                      # all splits -> eda/hist_features.csv
    python src/extract_hist_features.py --split test         # one split to stdout summary
    python src/extract_hist_features.py --verify             # recompute a few rows and
                                                             # compare against existing CSV

Output columns (match eda/hist_features.csv):
    rel_path, file_name, split, cls, ratio, pile_id, illum, roi_ok, hR0..hR31, hG0..hG31, hB0..hB31
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
BINS = 32
HIST_COLS = ([f"hR{i}" for i in range(BINS)]
             + [f"hG{i}" for i in range(BINS)]
             + [f"hB{i}" for i in range(BINS)])

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent


def masked_rgb_hist(img_bgr, mask=None, bins=BINS):
    """Return a length-3*bins vector: per-channel R,G,B masked, L1-normalized histograms.

    If mask is None it is computed with find_sample_roi. Returns (vector, roi_ok).
    Returns (None, ok) if the mask has too few pixels to be meaningful.
    """
    ok = True
    if mask is None:
        mask, _bbox, ok = find_sample_roi(img_bgr)
    valid = mask > 0
    if valid.sum() < 100:
        return None, ok
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    out = []
    for chan in (r, g, b):  # order R, G, B to match column naming
        h, _ = np.histogram(chan[valid], bins=bins, range=(0, 256))
        s = h.sum()
        out.append(h / s if s > 0 else h.astype(float))
    return np.concatenate(out).astype(float), ok


def ratio_from_path(rel: Path):
    """'adulterated/ratio_95_5/IMG (21).jpg' -> '95_5'; 'pure/IMG (1).jpg' -> '100_0'."""
    parts = rel.parts
    for p in parts:
        if p.startswith("ratio_"):
            return p[len("ratio_"):]
    if parts and parts[0] == "pure":
        return "100_0"
    return ""


def cls_from_path(rel: Path):
    p = str(rel).lower()
    if "pure" in p:
        return 0
    if "adulterated" in p or "starch" in p:
        return 1
    return None


def process_split(split, meta, split_root, warn_missing=True):
    base = split_root / split
    if not base.exists():
        if warn_missing:
            print(f"  WARN: split folder {base} does not exist — skipping")
        return []
    files = [p for p in sorted(base.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    rows = []
    for i, path in enumerate(files, 1):
        rel = path.relative_to(base)
        cls = cls_from_path(rel)
        if cls is None:
            print(f"  WARN: can't infer label for {rel} — skipping")
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARN: unreadable {rel} — skipping")
            continue
        vec, ok = masked_rgb_hist(img)
        if vec is None:
            print(f"  WARN: empty mask for {path.name} — skipping")
            continue
        fname = path.name
        md = meta.loc[fname] if fname in meta.index else None
        rows.append(dict(
            rel_path=str(rel).replace("\\", "/"), file_name=fname, split=split,
            cls=cls, ratio=ratio_from_path(rel),
            pile_id=(md["pile_id"] if md is not None else ""),
            illum=(md["illumination_color"] if md is not None else ""),
            roi_ok=int(ok), **dict(zip(HIST_COLS, vec))))
        if i % 50 == 0:
            print(f"    {split}: processed {i}/{len(files)}")
    print(f"  split '{split}': {len(rows)} rows")
    return rows


def verify(existing_csv, split_root, meta, n=3):
    """Recompute a handful of existing rows and report max abs bin difference."""
    old = pd.read_csv(existing_csv)
    sample = old.sample(min(n, len(old)), random_state=0)
    max_diff = 0.0
    for _, r in sample.iterrows():
        path = split_root / r["split"] / r["rel_path"]
        img = cv2.imread(str(path))
        if img is None:
            print(f"  VERIFY: could not load {path}"); continue
        vec, _ = masked_rgb_hist(img)
        if vec is None:
            print(f"  VERIFY: empty mask {path}"); continue
        old_vec = r[HIST_COLS].values.astype(float)
        d = float(np.max(np.abs(vec - old_vec)))
        max_diff = max(max_diff, d)
        print(f"  VERIFY {r['rel_path']}: max|Δbin| = {d:.2e}")
    print(f"\n  Overall max bin difference vs existing CSV: {max_diff:.2e}")
    print("  (≈1e-6 or smaller = exact match; large = normalization/mask mismatch)")
    return max_diff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--split-root", default=str(ROOT / "data" / "split"))
    ap.add_argument("--metadata", default=str(ROOT / "data" / "metadata.csv"))
    ap.add_argument("--out", default=str(ROOT / "eda" / "hist_features.csv"))
    ap.add_argument("--verify", action="store_true",
                    help="Recompute a few rows and compare to existing --out; write nothing.")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype=str).fillna("").set_index("file_name")
    split_root = Path(args.split_root)

    if args.verify:
        verify(Path(args.out), split_root, meta)
        return

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    rows = []
    for sp in splits:
        rows += process_split(sp, meta, split_root)
    if not rows:
        sys.exit("ERROR: no histogram features extracted.")
    df = pd.DataFrame(rows, columns=["rel_path", "file_name", "split", "cls",
                                     "ratio", "pile_id", "illum", "roi_ok"] + HIST_COLS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nDone. {len(df)} rows -> {out}")
    print(f"  class balance: {df['cls'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
