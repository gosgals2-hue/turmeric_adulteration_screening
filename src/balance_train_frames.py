"""
balance_train_frames.py — Balance the TRAIN set to 50/50 pure:adulterated after frame extraction.

Long pure videos yield more frames than short adulterated ones, leaving train imbalanced even
after the flipped sampling. This trims the MAJORITY class's frames (never images, never the
minority class) down to an equal pure:adulterated total.

Two properties that preserve diversity and reversibility:
  * STRATIFIED BY PILE: the frames to remove are spread across the majority class's piles in
    proportion to each pile's size, so every pile keeps a proportional share (no pile is drained).
  * MOVED, NOT DELETED: trimmed frame JPGs are moved to data/frames_aside/<split>/ (structure
    preserved), and their rows are removed from eda/features_frames_train.csv. Nothing is lost;
    re-running extract_frames.py or moving files back restores them.

Run AFTER extract_frames.py, BEFORE assemble_features.py (which reads frames from disk).

    python src/balance_train_frames.py --dry-run    # show per-pile trim plan
    python src/balance_train_frames.py              # apply
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent
VEXT = (".mov", ".mp4", ".avi")
NAME = {0: "pure", 1: "adulterated"}


def jpg_for(frames_root, split, rel_path):
    """'pure/IMG (73).MOV#f00015' -> data/frames/<split>/pure/IMG (73)_f00015.jpg"""
    vid, fidx = rel_path.split("#f")
    vid = Path(vid)
    return frames_root / split / vid.parent / f"{vid.stem}_f{fidx}.jpg"


def stratified_drop(frames_of_class, n_drop, seed):
    """Choose n_drop row-indices spread across piles in proportion to pile size."""
    counts = frames_of_class.groupby("pile_id").size()
    raw = counts / counts.sum() * n_drop
    alloc = np.floor(raw).astype(int)
    rem = int(n_drop - alloc.sum())
    if rem > 0:  # largest-remainder: give the leftovers to the biggest fractional parts
        for pile in (raw - alloc).sort_values(ascending=False).index[:rem]:
            alloc[pile] += 1
    alloc = alloc.clip(upper=counts)  # never drop more than a pile has
    idx = []
    for pile, k in alloc.items():
        if k > 0:
            sub = frames_of_class[frames_of_class.pile_id == pile]
            idx += sub.sample(int(k), random_state=seed).index.tolist()
    return idx, alloc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-csv", default=str(ROOT / "eda" / "features_frames_train.csv"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "split" / "manifest.csv"))
    ap.add_argument("--frames-root", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--aside-dir", default=str(ROOT / "data" / "frames_aside"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fr = pd.read_csv(args.frames_csv)
    man = pd.read_csv(args.manifest, dtype=str).fillna("")
    man = man[man.split == args.split]
    man_img = man[~man.file_name.str.lower().str.endswith(VEXT)]
    img = man_img.pile_id.map(lambda p: 0 if str(p).startswith("pure") else 1).value_counts().to_dict()
    frc = fr.cls.value_counts().to_dict()
    total = {c: img.get(c, 0) + frc.get(c, 0) for c in (0, 1)}
    print(f"train images: pure={img.get(0,0)} adul={img.get(1,0)}")
    print(f"train frames: pure={frc.get(0,0)} adul={frc.get(1,0)}")
    print(f"train TOTAL:  pure={total[0]} adul={total[1]}  (ratio {total[0]/max(total[1],1):.2f})")

    target = min(total.values())
    maj = max(total, key=total.get)
    n_drop = total[maj] - target
    if n_drop <= 0:
        print("Already balanced — nothing to do.")
        return
    drop_idx, alloc = stratified_drop(fr[fr.cls == maj], n_drop, args.seed)
    print(f"\nTrim {len(drop_idx)} {NAME[maj]} frames, stratified across {int((alloc>0).sum())} piles:")
    for pile, k in alloc[alloc > 0].items():
        print(f"  {pile}: -{int(k)}")

    if args.dry_run:
        print(f"\nDRY RUN: would move {len(drop_idx)} JPGs to {args.aside_dir}. Re-run without --dry-run.")
        return

    aside = Path(args.aside_dir); frames_root = Path(args.frames_root)
    moved = missing = 0
    for rp in fr.loc[drop_idx, "rel_path"]:
        src = jpg_for(frames_root, args.split, rp)
        if src.exists():
            dst = aside / args.split / src.relative_to(frames_root / args.split)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst)); moved += 1
        else:
            missing += 1
    fr_bal = fr.drop(index=drop_idx)
    fr_bal.to_csv(args.frames_csv, index=False)
    frc2 = fr_bal.cls.value_counts().to_dict()
    total2 = {c: img.get(c, 0) + frc2.get(c, 0) for c in (0, 1)}
    print(f"\nMoved {moved} JPGs to {aside} ({missing} already absent); updated {args.frames_csv}.")
    print(f"NEW train TOTAL: pure={total2[0]} adul={total2[1]}  (balanced)")


if __name__ == "__main__":
    main()
