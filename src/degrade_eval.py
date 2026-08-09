"""
degrade_eval.py — Robustness / degradation study (THE CORE CONTRIBUTION). Run locally.

Question: under realistic image degradation, WHERE and WHY does low-cost RGB pre-screening
fail? We apply photometric degradations to the TEST set only, re-featurize, score with the
already-trained classifiers at a FIXED deployment threshold, and plot sensitivity-vs-severity
curves per adulteration ratio. The output is an operating envelope, not a hero number.

Design principles (all deliberate):
  * Degrade TEST only. Photometric augmentation is kept OUT of training (see augment_train.py).
    That separation is what makes this measurement clean: the model has never seen degraded
    conditions, so any drop is genuine sensitivity to the degradation, not memorized recovery.
  * FIXED threshold. The operating threshold is chosen once on CLEAN val (as in deployment)
    and held constant across all conditions. You cannot re-tune per lighting in the field.
  * Deterministic models. Tier-1 (LogReg) and Tier-2 (SVM) are re-fit with the exact
    hyperparameters/seed of train_tier1_logreg.py / train_tier2_svm.py, so they reproduce the
    reported models without needing saved weights. Tier-3 (CNN) is loaded from
    experiments/exp_003_cnn/model_balanced.keras if present (--tiers cnn).
  * Whole-pipeline realism. By default the sample is re-segmented on the DEGRADED image
    (--mask redetect), so segmentation failures count as failures — that's the deployment
    truth. Use --mask clean to isolate the classifier from the segmenter (ablation).

Degradations & severity grids (0 = the clean baseline is always included):
  brightness  factor in {0.8, 0.9, 1.0, 1.1, 1.2}          (±20% exposure)
  contrast    factor in {0.8, 0.9, 1.0, 1.1, 1.2}          (±20% contrast about the mean)
  noise       gaussian sigma in {0, 5, 10, 15, 20, 25}     (added on 0-255 scale)
  blur        gaussian kernel in {1, 3, 5, 7, 9}           (1 = no blur)
  downscale   scale in {1.0, 0.75, 0.5, 0.35, 0.25}        (downscale then upscale back)
  jpeg        quality in {100, 75, 50, 30, 15}             (recompression artifacts)

Outputs (experiments/exp_degradation/):
  degradation_long.csv    tidy: tier,degradation,severity,ratio,n,detected,rate,metric,ci_lo,ci_hi
  curves_<degradation>.png sensitivity-vs-severity, one line per ratio (+ specificity on pure)
  heatmap_<tier>.png      ratio x degradation-severity -> sensitivity
  summary.md              per-tier "severity at which sensitivity falls below 0.90" table

REQUIREMENTS: pip install opencv-python numpy pandas scikit-learn matplotlib
              (+ tensorflow only if --tiers includes cnn)

RUN (from project root):
  python src/degrade_eval.py                          # tier1+tier2, all degradations
  python src/degrade_eval.py --tiers tier2            # just the best classical model
  python src/degrade_eval.py --degradations noise blur --quick
  python src/degrade_eval.py --tiers tier1 tier2 cnn  # include Tier-3 if trained
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi
from extract_hist_features import masked_rgb_hist, HIST_COLS

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FEATURES = ["mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "mean_V",
            "hue_std", "sat_std", "val_std", "bright", "contrast"]

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent
EXP = ROOT / "experiments" / "exp_degradation"

SEVERITY = {
    "brightness": [0.8, 0.9, 1.0, 1.1, 1.2],
    "contrast":   [0.8, 0.9, 1.0, 1.1, 1.2],
    "noise":      [0, 5, 10, 15, 20, 25],
    "blur":       [1, 3, 5, 7, 9],
    "downscale":  [1.0, 0.75, 0.5, 0.35, 0.25],
    "jpeg":       [100, 75, 50, 30, 15],
}
BASELINE = {"brightness": 1.0, "contrast": 1.0, "noise": 0, "blur": 1,
            "downscale": 1.0, "jpeg": 100}


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #
def degrade(img, kind, level, rng):
    if kind == "brightness":
        return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "contrast":
        m = img.mean()
        return np.clip((img.astype(np.float32) - m) * level + m, 0, 255).astype(np.uint8)
    if kind == "noise":
        if level == 0:
            return img
        n = rng.normal(0, level, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if kind == "blur":
        if level <= 1:
            return img
        k = int(level) | 1  # force odd
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "downscale":
        if level >= 1.0:
            return img
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * level)), max(1, int(h * level))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        if level >= 100:
            return img
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Featurization (11 stats + 96 hist share ONE mask)
# --------------------------------------------------------------------------- #
def masked_stats(img, mask):
    v = mask > 0
    if v.sum() < 100:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return [float(np.mean(r[v])), float(np.mean(g[v])), float(np.mean(b[v])),
            float(np.mean(H[v])), float(np.mean(S[v])), float(np.mean(V[v])),
            float(np.std(H[v])), float(np.std(S[v])), float(np.std(V[v])),
            float(np.mean(gray[v])), float(np.std(gray[v]))]


def featurize(img, clean_mask, mask_mode):
    """Return (feat11 or None, hist96 or None). mask_mode: 'redetect' or 'clean'."""
    if mask_mode == "clean" and clean_mask is not None:
        mask = clean_mask
    else:
        mask, _bbox, _ok = find_sample_roi(img)
    stats = masked_stats(img, mask)
    hist, _ = masked_rgb_hist(img, mask=mask)
    return stats, hist


# --------------------------------------------------------------------------- #
# Models (deterministic re-fit, matching the train scripts)
# --------------------------------------------------------------------------- #
def fit_tier1():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    tv = pd.read_csv(ROOT / "eda/features_trainval.csv")
    tr, va = tv[tv.split == "train"], tv[tv.split == "val"]
    sc = StandardScaler().fit(tr[FEATURES].values)
    clf = LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
    clf.fit(sc.transform(tr[FEATURES].values), tr.cls.values)

    def score_feat(f11, _h):
        if f11 is None:
            return None
        return float(clf.predict_proba(sc.transform([f11]))[0, 1])

    pva = clf.predict_proba(sc.transform(va[FEATURES].values))[:, 1]
    return score_feat, va.cls.values, pva


def fit_tier2():
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    df = pd.read_csv(ROOT / "eda/hist_features.csv")
    hfeats = [c for c in df.columns if c.startswith("h") and c[1] in "RGB"]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    sc = StandardScaler().fit(tr[hfeats].values)
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=SEED, class_weight="balanced")
    clf.fit(sc.transform(tr[hfeats].values), tr.cls.values)

    def score_hist(_f11, h96):
        if h96 is None:
            return None
        return float(clf.predict_proba(sc.transform([h96]))[0, 1])

    pva = clf.predict_proba(sc.transform(va[hfeats].values))[:, 1]
    return score_hist, va.cls.values, pva


def fit_cnn():
    """Load the trained Tier-3 PyTorch model; scorer takes an ROI-croppable BGR image."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2
    from torchvision import transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    mpath = ROOT / "experiments/exp_003_cnn/model_balanced.pt"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} not found — run train_tier3_cnn.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model.load_state_dict(torch.load(mpath, map_location=device))
    model.to(device).eval()
    tf_eval = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    vscores = pd.read_csv(ROOT / "experiments/exp_003_cnn/val_scores.csv")

    def score_cnn_crop(img_bgr, bbox):
        x, y, w, h = bbox                       # bbox from the shared segmentation (no re-seg)
        crop = img_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            crop = img_bgr
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            xb = tf_eval(rgb).unsqueeze(0).to(device)
            return float(torch.sigmoid(model(xb).ravel()).cpu().item())

    return score_cnn_crop, vscores.cls.values, vscores.score.values


def best_threshold(y, p):
    from sklearn.metrics import balanced_accuracy_score
    ts = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    best, bt = -1, 0.5
    for t in ts:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best:
            best, bt = ba, t
    return float(bt)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n; d = 1 + z * z / n; c = (ph + z * z / (2 * n)) / d
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - hw), min(1.0, c + hw)


# --------------------------------------------------------------------------- #
# Test-set enumeration
# --------------------------------------------------------------------------- #
def ratio_from_path(rel: Path):
    for p in rel.parts:
        if p.startswith("ratio_"):
            return p[len("ratio_"):]
    if rel.parts and rel.parts[0] == "pure":
        return "100_0"
    return ""


def enumerate_test(split_root, frames_root, use_frames, limit_per_ratio):
    base = split_root / "test"
    samples = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = path.relative_to(base)
            cls = 0 if "pure" in str(rel).lower() else 1
            samples.append(dict(path=str(path), cls=cls, ratio=ratio_from_path(rel)))
    if use_frames and (frames_root / "test").exists():
        for path in sorted((frames_root / "test").rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rel = path.relative_to(frames_root / "test")
                cls = 0 if "pure" in str(rel).lower() else 1
                samples.append(dict(path=str(path), cls=cls, ratio=ratio_from_path(rel)))
    if limit_per_ratio:  # smoke-test subsampling
        by = {}
        out = []
        for s in samples:
            by.setdefault(s["ratio"], 0)
            if by[s["ratio"]] < limit_per_ratio:
                out.append(s); by[s["ratio"]] += 1
        samples = out
    return samples


def rlabel(r):
    return "pure" if r in ("100_0", "") else r.replace("_", "/")


def rkey(r):
    try:
        return -float(r.split("_")[0])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=str(ROOT / "data" / "split"))
    ap.add_argument("--frames-root", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--tiers", nargs="+", default=["tier1", "tier2"],
                    choices=["tier1", "tier2", "cnn"])
    ap.add_argument("--degradations", nargs="+", default=list(SEVERITY.keys()),
                    choices=list(SEVERITY.keys()))
    ap.add_argument("--mask", default="redetect", choices=["redetect", "clean"],
                    help="redetect: re-segment degraded image (deployment-realistic). "
                         "clean: reuse the clean-image mask (isolates classifier).")
    ap.add_argument("--target-sp", type=float, default=None,
                    help="If set, fix the deployment threshold at this target specificity on "
                         "clean val (quantile of pure val scores). Else use max-bal-acc val thr.")
    ap.add_argument("--frames", dest="frames", action="store_true", default=True)
    ap.add_argument("--no-frames", dest="frames", action="store_false")
    ap.add_argument("--quick", action="store_true",
                    help="Subsample to 4 test items per ratio for a fast smoke test.")
    ap.add_argument("--limit-per-ratio", type=int, default=0,
                    help="Cap test items per ratio (0=all). A representative subsample keeps the "
                         "degradation curves meaningful while making the full run feasible.")
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="Ignore the checkpoint and process every item (default: resume).")
    ap.add_argument("--force", action="store_true", help="Delete the checkpoint and restart clean.")
    args = ap.parse_args()

    EXP.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- models + fixed deployment thresholds (from clean val) ----
    scorers = {}   # tier -> callable
    thr = {}       # tier -> float
    cnn_scorer = None
    if "tier1" in args.tiers:
        s1, yv1, pv1 = fit_tier1(); scorers["tier1"] = s1
        thr["tier1"] = (float(np.quantile(pv1[yv1 == 0], args.target_sp, method="higher"))
                        if args.target_sp else best_threshold(yv1, pv1))
    if "tier2" in args.tiers:
        s2, yv2, pv2 = fit_tier2(); scorers["tier2"] = s2
        thr["tier2"] = (float(np.quantile(pv2[yv2 == 0], args.target_sp, method="higher"))
                        if args.target_sp else best_threshold(yv2, pv2))
    if "cnn" in args.tiers:
        cnn_scorer, yvc, pvc = fit_cnn()
        thr["cnn"] = (float(np.quantile(pvc[yvc == 0], args.target_sp, method="higher"))
                      if args.target_sp else best_threshold(yvc, pvc))
    print("Fixed deployment thresholds (clean val):", {k: round(v, 3) for k, v in thr.items()})

    limit = args.limit_per_ratio or (4 if args.quick else 0)
    samples = enumerate_test(Path(args.split_root), Path(args.frames_root), args.frames, limit)
    print(f"Test samples: {len(samples)} "
          f"(pure {sum(s['cls']==0 for s in samples)} / adul {sum(s['cls']==1 for s in samples)})")

    # ---- resumable raw-record checkpoint ----
    raw_path = EXP / "_raw_records.csv"
    if args.force and raw_path.exists():
        raw_path.unlink(); print("  --force: cleared checkpoint")
    done_items = set()
    if args.resume and raw_path.exists():
        try:
            done_items = set(pd.read_csv(raw_path, usecols=["item"])["item"].astype(str).unique())
            print(f"  resume: {len(done_items)} item(s) already done — skipping them")
        except Exception:
            pass

    # ---- sweep: ONE segmentation per degraded image (shared by all tiers); flush per item ----
    for si, s in enumerate(samples):
        if s["path"] in done_items:
            continue
        img0 = cv2.imread(s["path"])
        if img0 is None:
            continue
        clean_mask = clean_bbox = None
        if args.mask == "clean":
            clean_mask, clean_bbox, _ = find_sample_roi(img0)
        item_rows = []
        for kind in args.degradations:
            for lvl in SEVERITY[kind]:
                dimg = degrade(img0, kind, lvl, rng)
                if args.mask == "clean" and clean_mask is not None:
                    mask, bbox = clean_mask, clean_bbox
                else:
                    mask, bbox, _ = find_sample_roi(dimg)     # segment once, reuse for all tiers
                f11 = masked_stats(dimg, mask) if scorers else None
                h96 = (masked_rgb_hist(dimg, mask=mask)[0]) if scorers else None
                for tier in args.tiers:
                    sc = cnn_scorer(dimg, bbox) if tier == "cnn" else scorers[tier](f11, h96)
                    pred = int(sc >= thr[tier]) if sc is not None else None
                    item_rows.append(dict(item=s["path"], tier=tier, degradation=kind,
                                          severity=lvl, ratio=s["ratio"], cls=s["cls"], pred=pred))
        pd.DataFrame(item_rows).to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
        if (si + 1) % 10 == 0:
            print(f"  processed {si + 1}/{len(samples)} test items")

    rec = pd.read_csv(raw_path, low_memory=False)
    rec_valid = rec.dropna(subset=["pred"]).copy()
    rec_valid["pred"] = rec_valid["pred"].astype(int)

    # ---- aggregate: per (tier,degradation,severity,ratio) sensitivity/specificity ----
    long_rows = []
    for (tier, kind, lvl, ratio), g in rec_valid.groupby(["tier", "degradation", "severity", "ratio"]):
        cls = g["cls"].iloc[0]
        n = len(g)
        if cls == 1:            # adulterated -> sensitivity = fraction flagged
            k = int((g["pred"] == 1).sum()); metric = "sensitivity"
        else:                   # pure -> specificity = fraction NOT flagged
            k = int((g["pred"] == 0).sum()); metric = "specificity"
        lo, hi = wilson(k, n)
        long_rows.append(dict(tier=tier, degradation=kind, severity=lvl, ratio=rlabel(ratio),
                              n=n, correct=k, rate=round(k / n, 4), metric=metric,
                              ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
    long = pd.DataFrame(long_rows)
    long.to_csv(EXP / "degradation_long.csv", index=False)
    print(f"  wrote {len(long)} aggregated rows -> {EXP/'degradation_long.csv'}")

    # ---- per-degradation curves (one line per ratio; pure shown as specificity) ----
    ratios_sorted = sorted(long["ratio"].unique(), key=lambda r: rkey("100_0" if r == "pure" else r.replace("/", "_")))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(ratios_sorted)))
    for tier in args.tiers:
        for kind in args.degradations:
            sub = long[(long.tier == tier) & (long.degradation == kind)]
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5.5))
            for ci, r in enumerate(ratios_sorted):
                rs = sub[sub.ratio == r].sort_values("severity")
                if rs.empty:
                    continue
                style = "--" if r == "pure" else "-"
                plt.plot(rs.severity, rs.rate, style, marker="o", color=cmap[ci], label=r, alpha=0.85)
            plt.axhline(0.9, ls=":", color="red", lw=1, label="0.90 reliability")
            plt.axvline(BASELINE[kind], ls=":", color="gray", lw=1)
            plt.ylim(0, 1.05); plt.xlabel(f"{kind} severity"); plt.ylabel("sensitivity (pure: specificity)")
            plt.title(f"{tier} — sensitivity vs {kind} (test; fixed thr={thr[tier]:.2f}; mask={args.mask})")
            plt.legend(fontsize=7, ncol=2, title="ratio")
            plt.tight_layout(); plt.savefig(EXP / f"curves_{tier}_{kind}.png", dpi=150); plt.close()

    # ---- heatmap per tier: ratio x (degradation@most-severe) sensitivity ----
    for tier in args.tiers:
        adul = long[(long.tier == tier) & (long.metric == "sensitivity")]
        if adul.empty:
            continue
        # define "worst" as the severity level furthest from the clean baseline
        worst = {}
        for k, v in SEVERITY.items():
            worst[k] = max(v, key=lambda x: abs(x - BASELINE[k]))
        degs = [d for d in args.degradations]
        rs = [r for r in ratios_sorted if r != "pure"]
        M = np.full((len(rs), len(degs)), np.nan)
        for j, d in enumerate(degs):
            for i, r in enumerate(rs):
                cell = adul[(adul.degradation == d) & (adul.ratio == r) & (adul.severity == worst[d])]
                if not cell.empty:
                    M[i, j] = cell["rate"].iloc[0]
        plt.figure(figsize=(1.4 * len(degs) + 2, 0.5 * len(rs) + 2))
        im = plt.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, label="sensitivity")
        plt.xticks(range(len(degs)), [f"{d}\n@{worst[d]}" for d in degs], fontsize=8)
        plt.yticks(range(len(rs)), rs, fontsize=8)
        for i in range(len(rs)):
            for j in range(len(degs)):
                if not np.isnan(M[i, j]):
                    plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.title(f"{tier} — sensitivity at worst severity (mask={args.mask})")
        plt.tight_layout(); plt.savefig(EXP / f"heatmap_{tier}.png", dpi=150); plt.close()

    # ---- summary.md: severity at which overall sensitivity first drops below 0.90 ----
    md = ["# Degradation study - summary\n",
          f"Mask mode: **{args.mask}**. Thresholds fixed on clean val "
          f"({', '.join(f'{k}={v:.2f}' for k,v in thr.items())}).",
          f"Test items: {len(samples)}. Photometric aug kept OUT of training.\n",
          "## Overall sensitivity (all adulterated ratios pooled) by severity\n"]
    for tier in args.tiers:
        md.append(f"\n### {tier}\n")
        md.append("| degradation | baseline | first severity with Se<0.90 | Se at worst severity |")
        md.append("|---|---|---|---|")
        for kind in args.degradations:
            g = rec_valid[(rec_valid.tier == tier) & (rec_valid.degradation == kind) & (rec_valid.cls == 1)]
            if g.empty:
                continue
            per = g.groupby("severity")["pred"].mean()
            base = BASELINE[kind]
            # order severities by distance from baseline
            order = sorted(per.index, key=lambda x: abs(x - base))
            first_break = "none"
            for lvl in order:
                if lvl == base:
                    continue
                if per[lvl] < 0.90:
                    first_break = str(lvl); break
            worst_lvl = max(per.index, key=lambda x: abs(x - base))
            md.append(f"| {kind} | {base} | {first_break} | {per[worst_lvl]:.2f} |")
    (EXP / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nDone. Figures + summary in {EXP}")
    print("  See summary.md for the 'severity at which Se falls below 0.90' table.")


if __name__ == "__main__":
    main()
