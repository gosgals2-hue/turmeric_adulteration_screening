"""
assemble_features.py — Build ALL classical feature tables for the current split, from disk.

This is the single, reproducible source for the tables the classical tiers read. It replaces
the previously ad-hoc (unscripted) generation of eda/features_trainval.csv, features_test.csv,
and hist_features.csv, and — critically — it includes the extracted video FRAMES alongside the
images, so Tier-1 and Tier-2 actually benefit from the frames (previously only the CNN did).

For every split image (data/split/<split>/) AND every extracted frame (data/frames/<split>/),
it computes, over ONE illumination-independent mask:
  - the 11 masked colour/texture stats (mean_R..contrast), and
  - the 96-d masked 32-bin RGB histogram (hR0..hB31).
Class, ratio, pile_id, and illumination are looked up from data/metadata.csv by the source
file's stem (frames map back to their parent video), so labels are correct even for the new
pure folders whose paths don't contain 'pure'.

Because it reads frames from disk, it automatically respects balance_train_frames.py (removed
frames are simply absent). Run AFTER split_dataset.py, extract_frames.py, and (optionally)
balance_train_frames.py.

Outputs:
  eda/features_trainval.csv   (train+val rows; 11 stats)   -> Tier-1, evaluate_protocol
  eda/features_test.csv       (test rows; 11 stats)        -> Tier-1, evaluate_protocol
  eda/hist_features.csv       (all rows; 96-d histogram)   -> Tier-2

Run from project root:
    python src/assemble_features.py                 # all splits
    python src/assemble_features.py --split test    # one split (appends nothing; prints only)
"""
import argparse
import re
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi
from extract_hist_features import masked_rgb_hist, HIST_COLS

IMG_EXT = {".jpg", ".jpeg", ".png"}
FEAT = ["mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "mean_V",
        "hue_std", "sat_std", "val_std", "bright", "contrast"]

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent


def masked_stats(img, mask):
    v = mask > 0
    if v.sum() < 100:
        return None, 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    stats = [float(np.mean(r[v])), float(np.mean(g[v])), float(np.mean(b[v])),
             float(np.mean(H[v])), float(np.mean(S[v])), float(np.mean(V[v])),
             float(np.std(H[v])), float(np.std(S[v])), float(np.std(V[v])),
             float(np.mean(gray[v])), float(np.std(gray[v]))]
    area_frac = float(v.sum()) / (img.shape[0] * img.shape[1])
    return stats, area_frac


VEXT = (".mov", ".mp4", ".avi")


def build_lookups(meta):
    """Return (by_file, by_vidstem). Images match by FULL filename (unique); frames match to
    their parent VIDEO by stem among video rows only. A bare-stem key is unsafe because the
    dataset reuses IMG numbers across categories (IMG (49).jpg is yellow-dye while IMG (49).MOV
    is starch), so it would collide and mislabel files."""
    by_file, by_vidstem = {}, {}
    for fn, row in meta.iterrows():
        pid = str(row["pile_id"])
        rec = dict(cls=0 if pid.startswith("pure") else 1, ratio=row.get("ratio", ""),
                   pile_id=pid, illum=row.get("illumination_color", ""),
                   illum_method=row.get("lighting_method", ""))
        by_file[fn] = rec
        if Path(fn).suffix.lower() in VEXT:
            by_vidstem[Path(fn).stem] = rec
    return by_file, by_vidstem


def orig_stem(name):
    return re.sub(r"_f\d+$", "", Path(name).stem)


def collect(split, split_root, frames_root, by_file, by_vidstem):
    rows = []
    sources = [("image", split_root / split), ("frame", frames_root / split)]
    for src_kind, base in sources:
        if not base.exists():
            continue
        files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
        for i, p in enumerate(files, 1):
            md = by_vidstem.get(orig_stem(p.name)) if src_kind == "frame" else by_file.get(p.name)
            if md is None:
                continue  # can't label -> skip
            img = cv2.imread(str(p))
            if img is None:
                continue
            mask, _bbox, ok = find_sample_roi(img)
            stats, area = masked_stats(img, mask)
            if stats is None:
                continue
            hist, _ = masked_rgb_hist(img, mask=mask)
            rows.append(dict(rel_path=str(p.relative_to(base)).replace("\\", "/"),
                             file_name=p.name, split=split, cls=md["cls"], ratio=md["ratio"],
                             pile_id=md["pile_id"], illum=md["illum"],
                             illum_method=md["illum_method"], roi_ok=int(ok), area_frac=round(area, 4),
                             source=src_kind,
                             **dict(zip(FEAT, stats)), **dict(zip(HIST_COLS, hist))))
            if i % 200 == 0:
                print(f"    {split}/{src_kind}: {i}/{len(files)}")
        print(f"  {split}/{src_kind}: {sum(1 for r in rows if r['source']==src_kind)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=str(ROOT / "data" / "split"))
    ap.add_argument("--frames-root", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--metadata", default=str(ROOT / "data" / "metadata.csv"))
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype=str).fillna("").set_index("file_name")
    by_file, by_vidstem = build_lookups(meta)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    allrows = []
    for sp in splits:
        print(f"=== {sp} ===")
        allrows += collect(sp, Path(args.split_root), Path(args.frames_root), by_file, by_vidstem)
    df = pd.DataFrame(allrows)
    if df.empty:
        sys.exit("ERROR: no features assembled.")

    stat_cols = ["rel_path", "file_name", "split", "cls", "ratio", "pile_id", "illum",
                 "illum_method", "source", "roi_ok", "area_frac"] + FEAT
    hist_cols = ["rel_path", "file_name", "split", "cls", "ratio", "pile_id", "illum",
                 "source", "roi_ok"] + HIST_COLS
    eda = ROOT / "eda"; eda.mkdir(exist_ok=True)

    if args.split == "all":
        df[df.split.isin(["train", "val"])][stat_cols].to_csv(eda / "features_trainval.csv", index=False)
        df[df.split == "test"][stat_cols].to_csv(eda / "features_test.csv", index=False)
        df[hist_cols].to_csv(eda / "hist_features.csv", index=False)
        print("\nWrote eda/features_trainval.csv, features_test.csv, hist_features.csv")

    print("\nrows per split x class x source:")
    print(df.assign(cls=df.cls.map({0: "pure", 1: "adul"}))
            .groupby(["split", "cls", "source"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
