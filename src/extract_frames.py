"""
extract_frames.py — Video frame extraction for the turmeric pipeline (run LOCALLY).

Reads data/split/manifest.csv and processes ONLY the requested split, pulling each
video from --data-root using the manifest's split assignment (videos are not copied
into the split folders). For every sampled frame it optionally (a) saves a JPG and
(b) computes the same 11 masked colour/texture features used by the image pipeline,
so the frames plug straight into training.

WHY asymmetric sampling (FLIPPED 2026-07-17): after adding 20 home-collected pure piles,
PURE is now the abundant class and ADULTERATED is scarce (adulterated samples are hard to
collect). So adulterated videos are sampled DENSELY (small interval) to maximize scarce
positive frames, and pure videos SPARSELY (large interval). VAL/TEST use a single uniform
interval and are NEVER augmented — they are evaluation data.

--------------------------------------------------------------------------------
REQUIREMENTS (install once):
    pip install opencv-python numpy pandas
    # scikit-learn is only needed later, for training — not for this script.
    # src/segment_bg.py must be present (it is imported for masking).

HOW TO RUN (from the project root, i.e. the folder containing data/ and src/):
    python src/extract_frames.py --split train      # dense pure / sparse adulterated
    python src/extract_frames.py --split val        # uniform, no augmentation
    python src/extract_frames.py --split test       # uniform, no augmentation
    python src/extract_frames.py --split all        # all three in one go
    python src/extract_frames.py --split train --dry-run   # plan only, no decoding

Outputs (defaults):
    data/frames/<split>/<original_subdirs>/<stem>_fNNNNN.jpg     (--save-frames)
    eda/features_frames_<split>.csv                              (--features)

The features CSV has the same columns as eda/features_trainval.csv plus source="frame"
and frame_idx, so you can concatenate:
    import pandas as pd
    img = pd.read_csv("eda/features_trainval.csv"); img["source"]="image"
    fr  = pd.read_csv("eda/features_frames_train.csv")
    train = pd.concat([img[img.split=="train"], fr], ignore_index=True)
--------------------------------------------------------------------------------
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi

VIDEO_EXT = (".mov", ".mp4", ".avi")
FEATURE_COLS = ["mean_R","mean_G","mean_B","mean_H","mean_S","mean_V",
                "hue_std","sat_std","val_std","bright","contrast"]


def interval_for(split, cls, args):
    """Frames are kept every Nth frame. Train uses asymmetric sampling; val/test uniform."""
    if split == "train":
        return args.pure_interval if str(cls) == "0" else args.adul_interval
    return args.uniform_interval


def masked_features(img):
    mask, _bbox, ok = find_sample_roi(img)
    v = mask > 0
    if v.sum() < 100:
        return None, ok
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]; H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    return [float(np.mean(r[v])), float(np.mean(g[v])), float(np.mean(b[v])),
            float(np.mean(H[v])), float(np.mean(S[v])), float(np.mean(V[v])),
            float(np.std(H[v])), float(np.std(S[v])), float(np.std(V[v])),
            float(np.mean(gray[v])), float(np.std(gray[v]))], ok


def already_done(out_csv):
    """Return the set of file_names already present in an existing per-split features CSV.

    Used for RESUME: video decoding is the expensive, kill-prone step, so we skip any
    video whose rows are already in the CSV. Granularity is per-video (a video is either
    fully in the CSV or not), which is safe because each video is flushed atomically.
    """
    if not out_csv.exists():
        return set()
    try:
        prev = pd.read_csv(out_csv, usecols=["file_name"])
        return set(prev["file_name"].astype(str).unique())
    except Exception:
        return set()


def process_split(split, man, meta, args):
    data_root = Path(args.data_root)
    vids = man[(man["split"] == split) &
               (man["rel_path"].str.lower().str.endswith(VIDEO_EXT))].reset_index(drop=True)
    if args.limit:
        vids = vids.iloc[:args.limit].reset_index(drop=True)
        print(f"  --limit {args.limit}: processing only the first {len(vids)} video(s) of this split")
    out = Path(args.out_features_dir) / f"features_frames_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.features and not args.dry_run:
        if args.force and out.exists():
            out.unlink(); print(f"  --force: removed existing {out}")
        elif args.resume:
            done = already_done(out)
            if done:
                print(f"  resume: {len(done)} video(s) already in {out.name} — will skip them")

    print(f"\n=== split '{split}': {len(vids)} videos ===")
    total_written = 0
    for vi, r in vids.iterrows():
        if args.features and not args.dry_run and r["file_name"] in done:
            print(f"  [{vi+1}/{len(vids)}] {r['rel_path']}  SKIP (already done)")
            continue
        src = data_root / r["rel_path"]
        if not src.exists():
            print(f"  MISSING: {r['rel_path']}"); continue
        step = interval_for(split, r["cls"], args)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            print(f"  UNREADABLE: {r['rel_path']}"); continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        md = meta.loc[r["file_name"]] if r["file_name"] in meta.index else None
        stem = Path(r["rel_path"]).stem; subdir = Path(r["rel_path"]).parent
        kept = 0
        if args.dry_run:
            tag = " (would SKIP, already done)" if r["file_name"] in already_done(out) else ""
            print(f"  [{vi+1}/{len(vids)}] {r['rel_path']}  cls={r['cls']} step={step} "
                  f"total_frames={total} -> ~{total//step} frames{tag}")
            cap.release(); continue
        recs = []  # per-video buffer, flushed atomically after the video decodes fully
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step == 0:
                if args.save_frames:
                    od = Path(args.out_frames) / split / subdir; od.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(od / f"{stem}_f{idx:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if args.features:
                    feats, roiok = masked_features(frame)
                    if feats is not None:
                        recs.append(dict(rel_path=f"{r['rel_path']}#f{idx:05d}", file_name=r["file_name"],
                            split=split, cls=int(r["cls"]), ratio=r["ratio"], pile_id=r["pile_id"],
                            illum=(md["illumination_color"] if md is not None else ""),
                            illum_method=(md["lighting_method"] if md is not None else ""),
                            **dict(zip(FEATURE_COLS, feats)), roi_ok=int(roiok),
                            source="frame", frame_idx=idx))
                kept += 1
            idx += 1
        cap.release()
        # Flush THIS video's rows immediately so an interrupted run keeps its progress.
        if args.features and recs:
            pd.DataFrame(recs).to_csv(out, mode="a", header=not out.exists(), index=False)
            total_written += len(recs)
        print(f"  [{vi+1}/{len(vids)}] {r['rel_path']}  cls={r['cls']} step={step}  "
              f"kept {kept} frames (flushed {len(recs)} rows)")
    if args.features and not args.dry_run:
        print(f"  {split}: appended {total_written} new frame-feature rows -> {out}")
    return total_written


def main():
    ap = argparse.ArgumentParser(description="Extract video frames per split from manifest.csv.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train","val","test","all"])
    ap.add_argument("--manifest", default="data/split/manifest.csv")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-frames", default="data/frames")
    ap.add_argument("--out-features-dir", default="eda")
    ap.add_argument("--pure-interval", type=int, default=50, help="Train: keep every Nth frame of PURE videos. Tuned so pure≈adulterated total (balanced).")
    ap.add_argument("--adul-interval", type=int, default=10, help="Train: keep every Nth frame of ADULTERATED videos (DENSE — scarce class, sampled maximally).")
    ap.add_argument("--uniform-interval", type=int, default=30, help="Val/Test: keep every Nth frame (no augmentation).")
    ap.add_argument("--save-frames", dest="save_frames", action="store_true", default=True)
    ap.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    ap.add_argument("--features", dest="features", action="store_true", default=True,
                    help="Also compute masked colour features per frame (default on).")
    ap.add_argument("--no-features", dest="features", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and projected frame counts; decode nothing.")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="Skip videos already present in features_frames_<split>.csv (default on). "
                         "Lets an interrupted run pick up where it left off.")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--force", action="store_true",
                    help="Delete any existing features_frames_<split>.csv and reprocess everything.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N videos per split (0 = all). Handy for a "
                         "quick local sanity check before committing to the full run.")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest, dtype=str).fillna("")
    meta = pd.read_csv(args.metadata, dtype=str).fillna("").set_index("file_name")
    splits = ["train","val","test"] if args.split == "all" else [args.split]
    print(f"Frame extraction | splits={splits} | save_frames={args.save_frames} | features={args.features}")
    print(f"  intervals: train pure=every {args.pure_interval}, train adul=every {args.adul_interval}, "
          f"val/test=every {args.uniform_interval}")
    total = 0
    for sp in splits:
        total += process_split(sp, man, meta, args)
    if not args.dry_run:
        print(f"\nDone. total frame-feature rows: {total}")


if __name__ == "__main__":
    main()
