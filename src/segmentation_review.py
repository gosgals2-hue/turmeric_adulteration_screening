"""
segmentation_review.py — Visual QA of segment_bg on EVERY image.

Read-only on the dataset. For each image it renders a 4-panel tile:
    [1] original   [2] binary mask   [3] original + mask/ROI overlay   [4] crop
saved under:  <out>/<group>/<lighting>/<stem>.jpg
where <group> is "pure" or "ratio_XX_XX" and <lighting> is the illumination
color from metadata (ambient / red / green / blue / yellow / cycling).

It scores each image's segmentation confidence (from find_sample_roi
diagnostics), writes a ranking CSV worst-first, and copies the lowest-confidence
/ near-fallback tiles into <out>/_hardest_cases/ (rank-prefixed) plus a montage,
so the hardest cases can be inspected first.

Run from project root:
    python src/segmentation_review.py                    # whole dataset + aggregate
    python src/segmentation_review.py --hardest 60 --include-starch
Batched (each batch fits a short time budget; aggregate once at the end):
    python src/segmentation_review.py --start 0   --end 60
    python src/segmentation_review.py --start 60  --end 120
    ...
    python src/segmentation_review.py --aggregate-only

The dataset is never modified or deleted — output goes only to <out>/.
"""

import argparse
import sys
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def risk_and_reason(ok, d):
    """Diagnostics -> (risk, confidence 0-100, near_fallback bool, reason str)."""
    reasons = []
    risk = 0.0
    if not ok:
        risk += 1.0
        reasons.append("FALLBACK (no blob found)")
    touches = d["touches"]
    if touches:
        risk += 0.15 * touches
        reasons.append(f"touches {touches} border(s)")
    sol = d["solidity"]
    if sol < 0.85:
        risk += 0.30 * (1.0 - sol)
        if sol < 0.7:
            reasons.append(f"ragged (solidity {sol:.2f})")
    cen = d["centeredness"]
    if cen < 0.6:
        risk += 0.15 * (1.0 - cen)
        reasons.append(f"off-center ({cen:.2f})")
    af = d["area_frac"]
    if af < 0.02:
        risk += 0.30 * min(1.0, (0.02 - af) / 0.02)
        reasons.append(f"tiny mask ({af*100:.1f}%)")
    elif af > 0.40:
        risk += 0.20 * min(1.0, (af - 0.40) / 0.60)
        reasons.append(f"huge mask ({af*100:.0f}%)")
    m = d["margin"]
    if np.isfinite(m):
        amb = max(0.0, min(1.0, (2.0 - m) / 1.0))
        if amb > 0:
            risk += 0.30 * amb
            if m < 1.3:
                reasons.append(f"ambiguous (margin {m:.2f})")
    near_fallback = (not ok) or touches >= 2 or (np.isfinite(m) and m < 1.2) \
        or af < 0.015 or sol < 0.6
    confidence = round(100.0 * float(np.exp(-risk)), 1)
    return risk, confidence, near_fallback, "; ".join(reasons) if reasons else "clean"


def _resize_h(im, h):
    s = h / im.shape[0]
    return cv2.resize(im, (max(1, int(im.shape[1] * s)), h))


def make_tile(img, mask, bbox, ok, header, panel_h):
    x, y, w, h = bbox
    green = np.zeros_like(img); green[:] = (0, 255, 0)
    overlay = img.copy()
    m3 = mask > 0
    overlay[m3] = (0.55 * img[m3] + 0.45 * green[m3]).astype(np.uint8)
    box_color = (0, 255, 0) if ok else (0, 165, 255)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, max(2, img.shape[1] // 400))
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        crop = np.zeros((panel_h, panel_h, 3), np.uint8)
    panels = [
        (_resize_h(img, panel_h), "original"),
        (_resize_h(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), panel_h), "mask"),
        (_resize_h(overlay, panel_h), "overlay" if ok else "overlay (FALLBACK)"),
        (_resize_h(crop, panel_h), "crop"),
    ]
    labeled = []
    for p, t in panels:
        cv2.rectangle(p, (0, 0), (p.shape[1], 26), (0, 0, 0), -1)
        col = (255, 255, 255) if "FALLBACK" not in t else (0, 165, 255)
        cv2.putText(p, t, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        labeled.append(p)
    row = np.hstack([cv2.copyMakeBorder(p, 0, panel_h - p.shape[0], 0, 4,
                                        cv2.BORDER_CONSTANT, value=(40, 40, 40))
                     for p in labeled])
    bar = np.zeros((30, row.shape[1], 3), np.uint8)
    cv2.putText(bar, header, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([bar, row])


def group_of(pile_id, ratio):
    if pile_id.startswith("pure"):
        return "pure"
    if pile_id.startswith("starch"):
        return "starch_" + ratio
    return "ratio_" + ratio


def aggregate(out_root, hardest_n):
    parts = sorted(glob.glob(str(out_root / "_rec_*.csv")))
    if not parts:
        sys.exit("No _rec_*.csv partial files found to aggregate.")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("rel_path")
    df = df.sort_values(["risk", "confidence"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(out_root / "confidence_ranking.csv", index=False)

    hard_dir = out_root / "_hardest_cases"
    hard_dir.mkdir(exist_ok=True)
    thumbs = []
    THUMB_W = 460
    for rank, (_, r) in enumerate(df.head(hardest_n).iterrows(), 1):
        stem = Path(r["file_name"]).stem
        tp = out_root / r["group"] / str(r["lighting"]) / f"{stem}.jpg"
        if not tp.exists():
            continue
        shutil.copy2(tp, hard_dir / f"{rank:03d}_conf{r['confidence']:.0f}_{stem}.jpg")
        tile = cv2.imread(str(tp))
        th = cv2.resize(tile, (THUMB_W, int(tile.shape[0] * THUMB_W / tile.shape[1])))
        cv2.rectangle(th, (0, 0), (48, 22), (0, 0, 200), -1)
        cv2.putText(th, f"#{rank}", (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        thumbs.append(th)
    if thumbs:
        cols = 3
        hmax = max(t.shape[0] for t in thumbs)
        thumbs = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in thumbs]
        rows = [np.hstack(thumbs[k:k + cols]) for k in range(0, len(thumbs), cols)]
        wmax = max(rr.shape[1] for rr in rows)
        rows = [cv2.copyMakeBorder(rr, 0, 0, 0, wmax - rr.shape[1],
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30)) for rr in rows]
        cv2.imwrite(str(hard_dir / "_montage.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"\n{'='*64}\n  AGGREGATE SUMMARY\n{'='*64}")
    print(f"  images ranked : {len(df)}")
    print(f"  fallbacks     : {int((df['ok']==0).sum())}")
    print(f"  near-fallback : {int(df['near_fallback'].sum())}")
    print(f"  confidence    : min={df['confidence'].min()} "
          f"median={df['confidence'].median()} mean={df['confidence'].mean():.1f}")
    print(f"\n  20 HARDEST (worst first):")
    for _, r in df.head(20).iterrows():
        print(f"    conf={r['confidence']:>5}  {r['group']:<15} {str(r['lighting']):<8} "
              f"{r['file_name']:<14} :: {r['reason']}")
    print(f"\n  Ranking CSV: {out_root}/confidence_ranking.csv")
    print(f"  Hardest cases + montage: {hard_dir}/")


def main():
    ap = argparse.ArgumentParser(description="Segmentation visual QA + hardest-case ranking.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--meta", default="data/metadata.csv")
    ap.add_argument("--out", default="segmentation_review")
    ap.add_argument("--hardest", type=int, default=45)
    ap.add_argument("--panel-h", type=int, default=340)
    ap.add_argument("--include-starch", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="Exclusive end index (0 = to end).")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip image processing; just build ranking/hardest from _rec_*.csv.")
    ap.add_argument("--aggregate", action="store_true",
                    help="After processing this batch, also build ranking/hardest.")
    args = ap.parse_args()

    data_root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(out_root, args.hardest)
        return

    meta = pd.read_csv(args.meta, dtype=str).fillna("")
    meta = meta[meta["file_name"].str.lower().str.endswith((".jpg", ".jpeg", ".png"))]
    if not args.include_starch:
        meta = meta[~meta["pile_id"].str.startswith("starch")]
    meta = meta.reset_index(drop=True)
    start = args.start
    end = args.end if args.end else len(meta)
    batch = meta.iloc[start:end]
    print(f"Batch rows [{start}:{end}] of {len(meta)} -> {out_root.resolve()}", flush=True)

    records = []
    n_fallback = 0
    for i, row in batch.iterrows():
        rel = row["folder"].rstrip("/") + "/" + row["file_name"]
        src = data_root / rel
        if not src.exists():
            print(f"  WARN missing: {rel}", flush=True); continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"  WARN unreadable: {rel}", flush=True); continue
        mask, bbox, ok, d = find_sample_roi(img, return_diagnostics=True)
        n_fallback += (not ok)
        risk, conf, near_fb, reason = risk_and_reason(ok, d)
        group = group_of(row["pile_id"], row["ratio"])
        lighting = row["illumination_color"] or "unknown"
        header = (f"{row['file_name']} | {group} | {lighting} | conf={conf} "
                  f"risk={risk:.2f} margin={d['margin']:.2f} touch={d['touches']} "
                  f"sol={d['solidity']:.2f} area={d['area_frac']*100:.1f}% | {reason}")
        tile = make_tile(img, mask, bbox, ok, header, args.panel_h)
        dst_dir = out_root / group / lighting
        dst_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_dir / f"{src.stem}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 88])
        records.append({
            "file_name": row["file_name"], "rel_path": rel, "group": group,
            "lighting": lighting, "lighting_method": row["lighting_method"],
            "pile_id": row["pile_id"], "ratio": row["ratio"],
            "ok": int(ok), "confidence": conf, "risk": round(risk, 3),
            "near_fallback": int(near_fb), "margin": round(d["margin"], 3),
            "touches": d["touches"], "area_frac": round(d["area_frac"], 4),
            "solidity": round(d["solidity"], 3), "centeredness": round(d["centeredness"], 3),
            "n_candidates": d["n_candidates"], "reason": reason,
        })

    pd.DataFrame(records).to_csv(out_root / f"_rec_{start:04d}_{end:04d}.csv", index=False)
    print(f"  batch done: {len(records)} tiles, {n_fallback} fallbacks", flush=True)

    if args.aggregate:
        aggregate(out_root, args.hardest)


if __name__ == "__main__":
    main()
