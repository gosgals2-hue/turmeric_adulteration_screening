"""
inspect_masks.py — Visually inspect HSV threshold masks before trusting them.

Run BEFORE editing extract_features.py or running augmentation on real data.
Review the saved images with your own eyes. If a mask looks wrong for any
image, tune the HSV bounds in this file and re-run until it looks right.

Run from project root:
    python src/inspect_masks.py --src data/pure --n 10
    python src/inspect_masks.py --src data/adulterated/ratio_90_10 --n 10
    python src/inspect_masks.py --src data --n 20   # sample across all classes

Output saved to:  mask_inspection/
    <image_name>_compare.jpg   <- side-by-side: original | mask | cropped result
    summary.txt                <- detection stats for every image inspected

What to look for:
    GOOD mask: covers the turmeric powder region, excludes white background
    BAD mask:  misses turmeric, or includes background/shadows as turmeric
    FALLBACK:  image name flagged as "center_crop_fallback" — means HSV failed

If you see lots of fallbacks or bad masks, adjust HSV_LOW and HSV_HIGH below.
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# HSV threshold bounds — TUNE THESE if masks look wrong
#
# OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
# Turmeric is yellow-orange: hue roughly 10-35 in OpenCV's scale
# Widen if you're missing turmeric; narrow if you're capturing background
# ---------------------------------------------------------------------------
HSV_LOW  = (8,  40,  40)
HSV_HIGH = (38, 255, 255)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_ROI_FRACTION = 0.15   # ROI must cover at least this fraction of image area


def find_roi(img):
    """
    Apply HSV threshold and return bounding box (x, y, w, h) of turmeric region.
    Returns (roi, used_fallback) where used_fallback=True means color detection failed.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]
    used_fallback = False

    if contours:
        all_pts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_pts)

        # Add margin
        margin = 8
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(W - x, w + 2 * margin)
        h = min(H - y, h + 2 * margin)

        if w * h >= MIN_ROI_FRACTION * W * H:
            return (x, y, w, h), mask, False

    # Fallback: center 70% of the image
    used_fallback = True
    mx, my = int(0.15 * W), int(0.15 * H)
    x, y = mx, my
    w, h = W - 2 * mx, H - 2 * my
    return (x, y, w, h), mask, True


def make_comparison(img, mask, roi, used_fallback):
    """
    Build a side-by-side comparison image:
      Left:   original with ROI bounding box drawn
      Middle: HSV mask (white = detected turmeric)
      Right:  cropped ROI result
    All panels are resized to the same height for display.
    """
    H, W = img.shape[:2]
    PANEL_H = 400
    scale = PANEL_H / H
    panel_w = int(W * scale)

    x, y, w, h = roi

    # Panel 1: original + bounding box
    panel1 = cv2.resize(img.copy(), (panel_w, PANEL_H))
    sx, sy = int(x * scale), int(y * scale)
    sw, sh = int(w * scale), int(h * scale)
    color = (0, 255, 0) if not used_fallback else (0, 165, 255)  # green or orange
    cv2.rectangle(panel1, (sx, sy), (sx + sw, sy + sh), color, 3)
    label = "HSV detected" if not used_fallback else "FALLBACK (center crop)"
    cv2.putText(panel1, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Panel 2: mask as BGR
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel2 = cv2.resize(mask_bgr, (panel_w, PANEL_H))
    cv2.putText(panel2, "HSV mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Panel 3: cropped ROI
    roi_img = img[y : y + h, x : x + w]
    roi_h, roi_w = roi_img.shape[:2]
    roi_scale = PANEL_H / roi_h if roi_h > 0 else 1
    roi_panel_w = max(1, int(roi_w * roi_scale))
    panel3 = cv2.resize(roi_img, (roi_panel_w, PANEL_H))
    cv2.putText(panel3, "Cropped ROI", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Stack horizontally with a thin divider
    divider = np.zeros((PANEL_H, 4, 3), dtype=np.uint8)
    comparison = np.hstack([panel1, divider, panel2, divider, panel3])
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HSV turmeric masks before running augmentation or feature extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", default="data",
                        help="Folder to sample images from (default: data/)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of images to inspect (default: 10)")
    parser.add_argument("--out", default="mask_inspection",
                        help="Output folder for comparison images (default: mask_inspection/)")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    if not src_root.exists():
        print(f"ERROR: Source folder '{src_root}' does not exist.")
        sys.exit(1)

    # Collect images
    all_images = sorted(
        f for f in src_root.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not all_images:
        print(f"ERROR: No images found under '{src_root}'.")
        sys.exit(1)

    # Sample evenly across the found images
    total = len(all_images)
    if total <= args.n:
        sampled = all_images
    else:
        step = total / args.n
        sampled = [all_images[int(i * step)] for i in range(args.n)]

    print(f"\nMask Inspection")
    print(f"  Source  : {src_root.resolve()}")
    print(f"  Sampling: {len(sampled)} of {total} images")
    print(f"  Output  : {out_root.resolve()}")
    print(f"  HSV low : {HSV_LOW}")
    print(f"  HSV high: {HSV_HIGH}\n")

    out_root.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "file | detected | fallback | turmeric_pct | roi_x | roi_y | roi_w | roi_h"
    ]

    fallback_count = 0

    for i, img_path in enumerate(sampled, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i}/{len(sampled)}] SKIP (could not read): {img_path.name}")
            continue

        H, W = img.shape[:2]
        (x, y, w, h), mask, used_fallback = find_roi(img)

        turmeric_pct = (mask > 0).sum() / (H * W) * 100
        if used_fallback:
            fallback_count += 1

        status = "FALLBACK" if used_fallback else "OK     "
        print(f"  [{i}/{len(sampled)}] {status}  {img_path.name}  "
              f"turmeric={turmeric_pct:.1f}%  roi=({x},{y},{w},{h})")

        comparison = make_comparison(img, mask, (x, y, w, h), used_fallback)
        out_name = f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_root / out_name), comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])

        summary_lines.append(
            f"{img_path.name} | {'yes' if not used_fallback else 'no'} | "
            f"{used_fallback} | {turmeric_pct:.1f}% | {x} | {y} | {w} | {h}"
        )

    # Save summary
    summary_path = out_root / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))

    # Report
    print(f"\n  Saved {len(sampled)} comparison images to: {out_root}/")
    print(f"  Fallbacks (HSV detection failed): {fallback_count}/{len(sampled)}")
    print(f"  Summary: {summary_path}")
    print()

    if fallback_count == 0:
        print("  HSV thresholds look good across all sampled images.")
        print("  Open the comparison images to visually confirm before proceeding.")
    elif fallback_count <= len(sampled) // 4:
        print(f"  {fallback_count} fallback(s) — acceptable, but inspect those images.")
        print("  Check if fallbacks are edge cases or a systematic failure.")
    else:
        print(f"  WARNING: {fallback_count}/{len(sampled)} images fell back to center crop.")
        print("  The HSV thresholds may not match your images.")
        print("  Adjust HSV_LOW and HSV_HIGH at the top of this script and re-run.")
        print()
        print("  Common causes:")
        print("    - Turmeric under colored SVI light shifts the hue (widen HSV_HIGH[0])")
        print("    - Dim lighting reduces saturation (lower HSV_LOW[1])")
        print("    - Background isn't white (tighten or change the threshold)")
    print()


if __name__ == "__main__":
    main()
