"""
segment_bg.py — Illumination-independent turmeric segmentation.

Replaces the HSV color-threshold placeholder. Works under any screen color
(ambient / red / green / blue / yellow) because it never assumes what color
the turmeric is. Logic:

    1. Sample the image BORDER (outer 4% band) -> median Lab color = background.
       The border is background in every shot; the mound is centered.
    2. Compute each pixel's distance from that background color in Lab space.
       Lab includes lightness, so a dark mound on a bright blue screen is
       "far" from background even when hue alone can't separate them.
    3. Otsu-threshold the distance map (auto picks the cut per image, so it
       adapts to how strong the contrast is under each lighting).
    4. Morphological open+close to drop speckle, then score components and
       keep the most plausible mound. Two-pass: strong border-touching
       distractors (fabric, shadows, table edges) are suppressed and the
       remainder re-thresholded.

Use as a module:
    from segment_bg import find_sample_roi
    mask, (x, y, w, h), ok = find_sample_roi(img)   # img = BGR uint8

Use as an inspection tool (trust your eyes before trusting the code):
    python src/segment_bg.py --src data/pure --n 6
    python src/segment_bg.py --src data/adulterated/ratio_90_10 --n 10
Saves side-by-side panels to mask_inspection_v2/.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Tunables (much fewer than the HSV version, and none are about color)
# ----------------------------------------------------------------------------
BORDER_FRAC = 0.04     # outer band sampled as "background" (4% of each edge)
WORK_WIDTH = 480       # downscale for speed; mask is rescaled back up
MIN_AREA_FRAC = 0.005  # component must cover >=0.5% of image to count
MORPH_OPEN_K = 15      # opening kernel: must be BIGGER than the wall/table
                       # junction line (~10px at 480 width) so a mound that
                       # overlaps the line gets separated from it, but far
                       # smaller than any mound (~80-150px)
MORPH_CLOSE_K = 7      # closing kernel: fill small holes inside the mound

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_sample_roi(img_bgr, return_diagnostics=False):
    """
    Locate the powder sample in a BGR image.

    Returns (mask, bbox, ok):
        mask : uint8 full-resolution binary mask (255 = sample)
        bbox : (x, y, w, h) of the chosen sample blob
        ok   : False if no plausible blob was found (mask/bbox are a
               center-crop fallback in that case)

    If return_diagnostics=True, returns (mask, bbox, ok, diag) where diag is a
    dict of confidence signals for the chosen blob (used by the segmentation
    review tool to rank the hardest cases). Keys:
        score        : winning candidate's raw score (higher = more mound-like)
        margin       : score / (best distinct runner-up score). ~1.0 means a
                       different blob almost won (ambiguous); large = decisive.
        touches      : how many image borders the chosen blob touches (0..4;
                       mounds shouldn't touch borders, distractors do)
        area_frac    : chosen mask area / image area
        solidity     : chosen blob area / its convex-hull area (compactness)
        centeredness : 1 - normalized distance of blob centroid from center
        n_candidates : number of scored candidates considered
    """
    H, W = img_bgr.shape[:2]
    scale = WORK_WIDTH / W
    small = cv2.resize(img_bgr, (WORK_WIDTH, int(H * scale)),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 1. background = the TWO dominant Lab colors of the border band.
    # Many shots have a two-tone background (table edge / wall line splits
    # the frame). A single median can't represent both, so the "other" tone
    # becomes a giant false blob. k=2 means table AND wall both count as
    # background; if the background is uniform the two centers just coincide.
    b = max(2, int(min(h, w) * BORDER_FRAC))
    border = np.concatenate([
        lab[:b, :].reshape(-1, 3), lab[-b:, :].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3),
    ]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, klabels, centers = cv2.kmeans(border, 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    # Guard: if the mound itself pokes into the border band, kmeans will
    # adopt IT as a "background" color and the mound becomes invisible.
    # A real second background tone (wall/table) covers a large share of
    # the border; a mound sliver does not. Only trust a center whose
    # cluster holds >=15% of border pixels.
    share = np.bincount(klabels.ravel(), minlength=4) / len(klabels)
    keep = [c for c, s in zip(centers, share) if s >= 0.10]
    if not keep:
        keep = [centers[int(np.argmax(share))]]

    # 2a. distance of every pixel from the NEAREST background color
    dist_border = np.minimum.reduce([np.linalg.norm(lab - c, axis=2) for c in keep])

    # 2b. LOCAL anomaly distance. Border colors fail when the mound happens
    # to match a distant background tone (e.g. dark-red mound vs the dim
    # half of a red-lit scene). A heavy median blur estimates what the
    # background looks like AT EACH PIXEL; the mound is whatever deviates
    # from its own surroundings. Complementary failure modes -> take max.
    tw = 160
    th = max(1, int(h * tw / w))
    tiny = cv2.resize(small, (tw, th), interpolation=cv2.INTER_AREA)
    lab_t = cv2.cvtColor(tiny, cv2.COLOR_BGR2LAB)
    # Gaussian pre-smooth: paper/fabric weave is a THREAD-scale anomaly that
    # would light up the whole textured area. We only care about MOUND-scale
    # anomalies, so average the threads away before comparing.
    lab_s = cv2.GaussianBlur(lab_t, (7, 7), 0)
    bg_est = cv2.merge([cv2.medianBlur(c, 71) for c in cv2.split(lab_s)])
    d_local = np.linalg.norm(lab_s.astype(np.float32) - bg_est.astype(np.float32), axis=2)
    d_local = cv2.resize(d_local, (w, h), interpolation=cv2.INTER_LINEAR)

    dist = np.maximum(dist_border, d_local)

    # 3+4. Candidate generation + scoring. Components are extracted at TWO
    # threshold levels (Otsu, and a stricter 1.5x Otsu that isolates only the
    # strongest anomaly = usually the mound core). Every component becomes a
    # candidate, scored by:
    #     dampened area   (big helps, but must not drown other cues)
    #   x border penalty  (distractors touch borders; mounds don't)
    #   x centeredness    (mound is framed near the middle)
    #   x solidity        (mounds are compact; ragged bg-texture blobs aren't)
    # Best-scoring candidate that doesn't touch 2+ borders wins.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    du8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu = cv2.threshold(du8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    candidates = []  # list of dicts (see below)
    cx0, cy0 = w / 2, h / 2
    diag = np.hypot(cx0, cy0)
    for level in (otsu, min(250.0, 1.5 * otsu), min(250.0, 2.2 * otsu)):
        _, mask = cv2.threshold(du8, level, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA_FRAC * h * w:
                continue
            x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            touches = (int(x <= 1) + int(y <= 1) +
                       int(x + bw >= w - 1) + int(y + bh >= h - 1))
            touch_penalty = {0: 1.0, 1: 0.5, 2: 0.15}.get(touches, 0.05)
            centeredness = 1.0 - np.hypot(cents[i][0] - cx0, cents[i][1] - cy0) / diag
            blob = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = max(1.0, cv2.contourArea(cv2.convexHull(
                max(cnts, key=cv2.contourArea))))
            solidity = min(1.0, area / hull_area)
            # cap area's influence: past ~8% of frame, bigger is not
            # more mound-like (distractor bands are huge; mounds are compact)
            eff_area = min(area, 0.08 * h * w)
            score = (eff_area ** 0.7) * touch_penalty * centeredness * (solidity ** 2)
            candidates.append({
                "score": score, "touches": touches, "blob": blob * 255,
                "bbox": (x, y, bw, bh), "area_frac": area / (h * w),
                "solidity": solidity, "centeredness": max(0.0, centeredness),
            })

    if not candidates:
        if return_diagnostics:
            mask, bbox, ok = _fallback(H, W)
            return mask, bbox, ok, {"score": 0.0, "margin": 0.0, "touches": 4,
                                    "area_frac": 0.25, "solidity": 0.0,
                                    "centeredness": 0.0, "n_candidates": 0}
        return _fallback(H, W)
    candidates.sort(key=lambda c: -c["score"])
    chosen = next((c for c in candidates if c["touches"] < 2), candidates[0])
    x, y, bw, bh = chosen["bbox"]
    blob = chosen["blob"]

    # margin: how decisively the winner beat the best DISTINCT alternative.
    # Distinct = a candidate whose bbox overlaps the winner's by < 50% (IoU),
    # so overlapping blobs from different threshold levels don't count as rivals.
    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw2, bh2 = b
        ix = max(0, min(ax + aw, bx + bw2) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh2) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw2 * bh2 - inter
        return inter / union if union > 0 else 0.0
    runner = next((c for c in candidates
                   if _iou(c["bbox"], chosen["bbox"]) < 0.5), None)
    margin = (chosen["score"] / runner["score"]) if (runner and runner["score"] > 0) \
        else float("inf")

    # fill holes: a powder mound has no holes; "donuts" happen when the
    # mound's core happens to match a background tone
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blob, cnts, -1, 255, thickness=-1)

    # rescale to full resolution
    full_mask = cv2.resize(blob, (W, H), interpolation=cv2.INTER_NEAREST)
    inv = 1.0 / scale
    bbox = (int(x * inv), int(y * inv), int(bw * inv), int(bh * inv))
    if return_diagnostics:
        diag_out = {
            "score": float(chosen["score"]),
            "margin": float(margin),
            "touches": int(chosen["touches"]),
            "area_frac": float(chosen["area_frac"]),
            "solidity": float(chosen["solidity"]),
            "centeredness": float(chosen["centeredness"]),
            "n_candidates": len(candidates),
        }
        return full_mask, bbox, True, diag_out
    return full_mask, bbox, True


def _fallback(H, W):
    """Center crop when detection fails - flagged so callers can skip/log."""
    x, y = W // 4, H // 4
    mask = np.zeros((H, W), np.uint8)
    mask[y:y + H // 2, x:x + W // 2] = 255
    return mask, (x, y, W // 2, H // 2), False


# ----------------------------------------------------------------------------
# Inspection CLI
# ----------------------------------------------------------------------------
def _panel(img, mask, bbox, ok, out_h=400):
    x, y, w, h = bbox
    vis = img.copy()
    color = (0, 255, 0) if ok else (0, 165, 255)
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, max(2, img.shape[1] // 400))
    label = "detected" if ok else "FALLBACK"

    def rs(im):
        s = out_h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), out_h))

    p1, p2 = rs(vis), rs(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    crop = img[y:y + h, x:x + w]
    p3 = rs(crop) if crop.size else np.zeros((out_h, out_h, 3), np.uint8)
    for p, t in ((p1, label), (p2, "bg-distance mask"), (p3, "cropped ROI")):
        cv2.rectangle(p, (0, 0), (p.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(p, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255) if t != "FALLBACK" else (0, 165, 255), 2)
    return np.hstack([p1, p2, p3])


def main():
    ap = argparse.ArgumentParser(description="Inspect background-difference masks.")
    ap.add_argument("--src", required=True, help="folder of images")
    ap.add_argument("--n", type=int, default=8, help="how many images to sample")
    ap.add_argument("--out", default="mask_inspection_v2")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.src).iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        sys.exit(f"No images found in {args.src}")
    step = max(1, len(files) // args.n)
    sample = files[::step][:args.n]

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fallbacks = 0
    for i, p in enumerate(sample, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{i}/{len(sample)}] unreadable: {p.name}")
            continue
        mask, bbox, ok = find_sample_roi(img)
        fallbacks += (not ok)
        frac = float(np.mean(mask > 0)) * 100
        print(f"  [{i}/{len(sample)}] {'OK      ' if ok else 'FALLBACK'} "
              f"{p.name}  sample={frac:.1f}%  roi={bbox}")
        cv2.imwrite(str(out / f"{p.stem}_v2.jpg"), _panel(img, mask, bbox, ok))

    print(f"\n  Saved {len(sample)} panels to {out}/  ({fallbacks} fallbacks)")


if __name__ == "__main__":
    main()
