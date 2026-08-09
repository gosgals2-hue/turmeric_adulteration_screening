"""
augment_train.py — Geometric augmentation for the train split only.

Run from project root:
    python src/augment_train.py
    python src/augment_train.py --src data/split/train --out data/split/train_aug
    python src/augment_train.py --grid 6 --frame-interval 60

Augmentations applied (geometric only):
    - Grid patches (4x4 or 6x6) extracted from the turmeric region
    - Horizontal and vertical flips
    - Rotations: +30, +15, -15, -30 degrees
    - Shear: horizontal and vertical (both directions)
    - Mild perspective warp (simulates camera tilt)

For videos: extracts one frame every --frame-interval frames, then applies
the full augmentation pipeline to each extracted frame.

WHY no color augmentation here:
    Brightness, contrast, and Gaussian noise are reserved for the degradation
    evaluation pipeline. Including them in training would contaminate the
    robustness experiment — the model would have already "seen" degraded
    conditions, making it impossible to cleanly measure how degradation hurts
    performance. Keep these two pipelines fully separate.
"""

import shutil
import cv2
import numpy as np
import argparse
import sys
import pandas as pd
from pathlib import Path

# segment_bg lives alongside this file in src/. Make the import work whether
# the script is run as `python src/augment_train.py` (cwd = project root) or
# imported as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

MIN_PATCH_PX = 48      # discard patches smaller than this (too small to be meaningful)
JPEG_QUALITY = 92      # saved image quality — high enough to not lose color detail


# ---------------------------------------------------------------------------
# ROI detection — illumination-independent segmentation (segment_bg)
# ---------------------------------------------------------------------------
# The old HSV find_turmeric_roi() assumed the powder was yellow-orange on a
# white background; that assumption breaks under SVI colored lighting (red /
# green / blue / yellow screens). segment_bg.find_sample_roi() locates the
# mound by its distance from the background color in Lab space, so it works
# under any illumination. It returns (mask, bbox, ok); ok=False means it fell
# back to a center crop, which we still crop to (same graceful-degradation
# behavior the old code had).

def extract_roi(img, grid_n):
    """
    Return the turmeric region as a cropped image, located with segment_bg.

    find_sample_roi() already supplies a sensible center-crop fallback (ok=False)
    when no plausible mound is found, so we simply crop to its bounding box.
    """
    _mask, (x, y, w, h), _ok = find_sample_roi(img)
    return img[y : y + h, x : x + w]


def grid_patches(roi_img, grid_n):
    """
    Divide roi_img into an (grid_n x grid_n) grid of patches.
    Returns list of (patch, row, col) — only patches large enough to be useful.

    Why grid patches? Your images contain a lot of white background that carries
    no signal. Cropping into small patches focuses the model on actual turmeric
    pixels and multiplies your training sample count significantly.
    """
    H, W = roi_img.shape[:2]
    cell_h = H // grid_n
    cell_w = W // grid_n

    patches = []
    for row in range(grid_n):
        for col in range(grid_n):
            py = row * cell_h
            px = col * cell_w
            patch = roi_img[py : py + cell_h, px : px + cell_w]
            if patch.shape[0] >= MIN_PATCH_PX and patch.shape[1] >= MIN_PATCH_PX:
                patches.append((patch, row, col))

    return patches


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

def augment(img):
    """
    Apply all geometric augmentations to img.
    Returns list of (name, augmented_image) tuples.
    Order matters for reproducibility — don't randomize.
    """
    H, W = img.shape[:2]
    cx, cy = W / 2, H / 2
    results = []

    results.append(("orig", img))

    # Flips
    results.append(("fh", cv2.flip(img, 1)))
    results.append(("fv", cv2.flip(img, 0)))

    # Rotations — BORDER_REFLECT_101 fills corners without introducing
    # a solid color, which would create false signal near patch edges
    for angle in [15, -15, 30, -30]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        label = f"r{angle:+d}".replace("+", "p").replace("-", "n")
        results.append((label, rotated))

    # Shear — horizontal (shifts left vs right columns relative to each other)
    for sx, label in [(0.12, "shx_p"), (-0.12, "shx_n")]:
        M = np.float32([[1, sx, 0], [0, 1, 0]])
        new_w = int(W + abs(sx) * H)
        results.append((label, cv2.warpAffine(img, M, (new_w, H),
                                               borderMode=cv2.BORDER_REFLECT_101)))

    # Shear — vertical
    for sy, label in [(0.12, "shy_p"), (-0.12, "shy_n")]:
        M = np.float32([[1, 0, 0], [sy, 1, 0]])
        new_h = int(H + abs(sy) * W)
        results.append((label, cv2.warpAffine(img, M, (W, new_h),
                                               borderMode=cv2.BORDER_REFLECT_101)))

    # Perspective warp — simulates slight camera tilt (~5 degrees off-axis)
    # The four corner shifts mimic what happens when a phone isn't held flat
    offset = int(min(H, W) * 0.05)
    for label, src_pts, dst_pts in [
        ("persp_tl", np.float32([[0,0],[W,0],[0,H],[W,H]]),
                     np.float32([[offset,offset],[W,0],[0,H],[W,H]])),
        ("persp_tr", np.float32([[0,0],[W,0],[0,H],[W,H]]),
                     np.float32([[0,0],[W-offset,offset],[0,H],[W,H]])),
    ]:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
        results.append((label, warped))

    return results


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def save_aug(img, out_dir, fname):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def process_image(src_path, out_dir, grid_n, include_full):
    """
    Process one image:
      1. Extract turmeric ROI
      2. Divide into grid_n x grid_n patches
      3. Augment each patch
      4. Also augment the full image (for feature-based models)
    Returns (count, list_of_log_dicts).
    """
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  WARN: Could not read {src_path.name} — skipping.")
        return 0, []

    stem = src_path.stem
    count = 0
    records = []

    roi_img = extract_roi(img, grid_n)
    patches = grid_patches(roi_img, grid_n)

    if not patches:
        print(f"  WARN: No valid patches from {src_path.name} (image may be too small).")

    for patch, row, col in patches:
        for aug_name, aug_img in augment(patch):
            fname = f"{stem}_p{row}{col}_{aug_name}.jpg"
            save_aug(aug_img, out_dir, fname)
            records.append({"source": src_path.name, "output": fname, "kind": "patch"})
            count += 1

    if include_full:
        for aug_name, aug_img in augment(img):
            fname = f"{stem}_full_{aug_name}.jpg"
            save_aug(aug_img, out_dir, fname)
            records.append({"source": src_path.name, "output": fname, "kind": "full"})
            count += 1

    return count, records


def process_video(src_path, out_dir, grid_n, include_full, frame_interval):
    """
    Extract frames from a video at every frame_interval-th frame,
    then apply the full augmentation pipeline to each frame.

    frame_interval=60 at 30fps → 1 frame every 2 seconds.
    Use a higher interval to keep sample count manageable.
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"  WARN: Could not open video {src_path.name} — skipping.")
        return 0, []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    expected_frames = total // frame_interval

    print(f"  Video: {src_path.name} | {fps:.0f} fps | {duration:.1f}s | "
          f"~{expected_frames} frames to extract")

    stem = src_path.stem
    frame_idx = 0
    extracted = 0
    count = 0
    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            roi_img = extract_roi(frame, grid_n)
            patches = grid_patches(roi_img, grid_n)

            for patch, row, col in patches:
                for aug_name, aug_img in augment(patch):
                    fname = f"{stem}_f{frame_idx:05d}_p{row}{col}_{aug_name}.jpg"
                    save_aug(aug_img, out_dir, fname)
                    records.append({"source": src_path.name, "output": fname, "kind": "video_patch"})
                    count += 1

            if include_full:
                for aug_name, aug_img in augment(frame):
                    fname = f"{stem}_f{frame_idx:05d}_full_{aug_name}.jpg"
                    save_aug(aug_img, out_dir, fname)
                    records.append({"source": src_path.name, "output": fname, "kind": "video_full"})
                    count += 1

            extracted += 1

        frame_idx += 1

    cap.release()
    print(f"    -> {extracted} frames extracted, {count} augmented samples saved")
    return count, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment turmeric train split (geometric transforms only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", default="data/split/train",
                        help="Source folder (default: data/split/train)")
    parser.add_argument("--out", default="data/split/train_aug",
                        help="Output folder (default: data/split/train_aug)")
    parser.add_argument("--grid", type=int, choices=[4, 6], default=4,
                        help="Grid size for patch extraction: 4 (4x4=16 patches) "
                             "or 6 (6x6=36 patches). Default: 4")
    parser.add_argument("--frame-interval", type=int, default=60,
                        help="Extract 1 frame every N frames from videos. "
                             "At 30fps: 60=every 2s, 30=every 1s. Default: 60")
    parser.add_argument("--no-full", action="store_true",
                        help="Skip full-image augmentations (patches only)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output folder")
    parser.add_argument("--manifest", default="data/split/manifest.csv",
                        help="Split manifest. Videos are NOT copied into the split "
                             "folders (too large); they are read from --data-root "
                             "using this manifest's split assignment. "
                             "Default: data/split/manifest.csv")
    parser.add_argument("--data-root", default="data",
                        help="Root of the original dataset where videos live "
                             "(default: data). Videos are sourced from here.")
    parser.add_argument("--split", default=None,
                        help="Which split's videos to pull from the manifest "
                             "(train/val/test). Default: inferred from --src folder name.")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    include_full = not args.no_full
    split_name = args.split or src_root.name   # e.g. .../split/train -> "train"

    print(f"\nTurmeric Augmentation Pipeline")
    print(f"  Source         : {src_root.resolve()}")
    print(f"  Output         : {out_root.resolve()}")
    print(f"  Grid size      : {args.grid}x{args.grid} = {args.grid**2} patches per image")
    print(f"  Frame interval : every {args.frame_interval} frames")
    print(f"  Full images    : {'yes' if include_full else 'no (patches only)'}")

    # orig(1) + fh(1) + fv(1) + rot×4(4) + shearH×2(2) + shearV×2(2) + persp×2(2) = 13
    per_patch = 13
    patches_per_img = args.grid ** 2
    per_image_est = patches_per_img * per_patch + (per_patch if include_full else 0)
    print(f"  Est. samples per image : ~{per_image_est} "
          f"({patches_per_img} patches × {per_patch} augmentations"
          + (f" + {per_patch} full" if include_full else "") + ")")

    if not src_root.exists():
        print(f"\n  ERROR: Source folder '{src_root}' does not exist.")
        print("  Run split_dataset.py first, then augment the train split.")
        sys.exit(1)

    if out_root.exists() and any(out_root.iterdir()):
        if not args.force:
            print(f"\n  ERROR: Output folder '{out_root}' already exists and is not empty.")
            print("  Add --force to overwrite it.")
            sys.exit(1)
        else:
            shutil.rmtree(out_root)
            print(f"  --force: cleared existing output.")

    # Images come from the split folder (they were copied there by split_dataset).
    images = sorted(f for f in src_root.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)

    # Videos are NOT copied into the split folders (they are large and the split
    # folders would balloon). Instead we read this split's videos straight from
    # --data-root using the manifest's split assignment. This keeps whole videos
    # as single units on the correct side of the split (no leakage) without
    # duplicating gigabytes of footage.
    videos = []   # list of (source_path, rel_path) pairs
    manifest_path = Path(args.manifest)
    data_root = Path(args.data_root)
    if manifest_path.exists():
        man = pd.read_csv(manifest_path, dtype=str).fillna("")
        vids = man[(man["split"] == split_name) &
                   (man["rel_path"].str.lower().str.endswith((".mov", ".mp4", ".avi")))]
        for _, r in vids.iterrows():
            src = data_root / r["rel_path"]
            if src.exists():
                videos.append((src, Path(r["rel_path"])))
            else:
                print(f"  WARN: manifest video missing on disk: {r['rel_path']}")
        print(f"\n  Split '{split_name}': {len(images)} images (from {src_root}) "
              f"and {len(videos)} videos (from {data_root} via manifest).")
    else:
        print(f"\n  WARN: manifest '{manifest_path}' not found — processing images "
              f"only, no video frames extracted.")
        print(f"  Found {len(images)} images.")

    if not images and not videos:
        print(f"\n  ERROR: No images under '{src_root}' and no videos in manifest.")
        sys.exit(1)

    # Process
    total_count = 0
    all_records = []

    section(f"IMAGES ({len(images)})")
    for i, src_path in enumerate(images, 1):
        rel = src_path.relative_to(src_root)
        out_dir = out_root / rel.parent
        print(f"  [{i}/{len(images)}] {rel}")
        n, records = process_image(src_path, out_dir, args.grid, include_full)
        total_count += n
        all_records.extend(records)

    if videos:
        section(f"VIDEOS ({len(videos)})")
        for i, (src_path, rel) in enumerate(videos, 1):
            out_dir = out_root / rel.parent
            print(f"  [{i}/{len(videos)}] {rel}")
            n, records = process_video(src_path, out_dir, args.grid,
                                       include_full, args.frame_interval)
            total_count += n
            all_records.extend(records)

    # Save log
    section("SAVING LOG")
    log_path = out_root / "augmentation_log.csv"
    pd.DataFrame(all_records).to_csv(log_path, index=False)
    print(f"  Log saved: {log_path}")

    # Summary
    section("SUMMARY")
    df_log = pd.DataFrame(all_records)
    if not df_log.empty:
        by_kind = df_log["kind"].value_counts()
        print(f"  Total augmented samples : {total_count}")
        for kind, cnt in by_kind.items():
            print(f"    {kind:<20} {cnt}")

    print(f"\n  Output at: {out_root.resolve()}")
    print(f"\n  NEXT STEP: Run your EDA script on the augmented train set.")
    print(f"  Val and test sets were NOT touched.")
    print()


if __name__ == "__main__":
    main()
