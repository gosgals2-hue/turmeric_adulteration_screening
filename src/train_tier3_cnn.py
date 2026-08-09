"""
train_tier3_cnn.py — Tier-3: MobileNetV2 transfer learning on ROI crops. RUN LOCALLY.

PyTorch / torchvision implementation (installs cleanly on modern Python incl. 3.13,
unlike TensorFlow which has no wheel there).

Completes the model-comparison table: does a pretrained CNN beat handcrafted colour
features (Tier-1 means, Tier-2 96-d histograms)? A CNN that ties or LOSES to histograms
is itself a valid finding (handcrafted colour is sufficient on this small dataset), so
this script reports honestly under the same protocol as Tier-1/2:

    train on TRAIN, choose threshold on VAL (max balanced-accuracy), report on sealed TEST;
    both unweighted and class-balanced; positive class = adulterated.

Inputs are ROI CROPS (segment_bg bbox) resized to 224x224, so the CNN sees the sample, not
the background — consistent with the masked classical features. Images come from the split
folders; video frames (if extract_frames.py --save-frames was run) are pulled from
data/frames/<split>/ and added to TRAIN/VAL/TEST. Photometric augmentation is deliberately
NOT applied here (it belongs to the degradation experiment); only light geometric aug is used
during training, matching the "no degraded conditions seen in training" separation.

Outputs (experiments/exp_003_cnn/):
    metrics.json                  same schema as Tier-1/2 (unweighted + balanced variants)
    test_scores.csv               per-test-sample scores (file_name,rel_path,pile_id,ratio,cls,score)
    val_scores.csv                per-val-sample scores (for threshold selection / auditing)
    model_<variant>.pt            trained weights (state_dict) for degrade_eval.py reuse
    pr_and_metrics.png            PR + bar chart
    comparison_all_tiers.md       Tier-1 vs Tier-2 vs Tier-3 table

REQUIREMENTS (install once, locally):
    pip install torch torchvision opencv-python numpy pandas scikit-learn matplotlib

HOW TO RUN (from project root):
    python src/train_tier3_cnn.py                     # images + frames if present
    python src/train_tier3_cnn.py --no-frames         # images only (quick smoke test)
    python src/train_tier3_cnn.py --epochs 15 --fine-tune
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_bg import find_sample_roi

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 224
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent
EXP = ROOT / "experiments" / "exp_003_cnn"


# --------------------------------------------------------------------------- #
# Sample enumeration (shared logic with the classical pipeline)
# --------------------------------------------------------------------------- #
def ratio_from_path(rel: Path):
    for p in rel.parts:
        if p.startswith("ratio_"):
            return p[len("ratio_"):]
    if rel.parts and rel.parts[0] == "pure":
        return "100_0"
    return ""


def cls_from_path(rel: Path):
    p = str(rel).lower()
    if "pure" in p:
        return 0
    if "adulterated" in p or "starch" in p:
        return 1
    return None


def build_stem_to_pile(meta):
    # VIDEO rows ONLY: the dataset reuses IMG numbers across categories (IMG (49).jpg vs
    # IMG (49).MOV), so keying on all stems would collide and mis-assign a frame's pile.
    out = {}
    for fname, row in meta.iterrows():
        if Path(fname).suffix.lower() in (".mov", ".mp4", ".avi"):
            out[Path(fname).stem] = row["pile_id"]
    return out


def enumerate_split(split, meta, stem2pile, split_root, frames_root, use_frames):
    samples = []
    base = split_root / split
    if base.exists():
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = path.relative_to(base)
            cls = cls_from_path(rel)
            if cls is None:
                continue
            md = meta.loc[path.name] if path.name in meta.index else None
            samples.append(dict(path=str(path), cls=cls, ratio=ratio_from_path(rel),
                                pile_id=(md["pile_id"] if md is not None else ""),
                                file_name=path.name, source="image"))
    if use_frames:
        fbase = frames_root / split
        if fbase.exists():
            for path in sorted(fbase.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel = path.relative_to(fbase)
                cls = cls_from_path(rel)
                if cls is None:
                    continue
                orig_stem = re.sub(r"_f\d+$", "", path.stem)
                pile = stem2pile.get(orig_stem, "")
                samples.append(dict(path=str(path), cls=cls, ratio=ratio_from_path(rel),
                                    pile_id=pile, file_name=path.name, source="frame"))
    return samples


def load_roi(path):
    """Load image, crop to segment_bg ROI bbox, resize to IMG_SIZE, return RGB uint8."""
    img = cv2.imread(path)
    if img is None:
        return None
    _mask, (x, y, w, h), _ok = find_sample_roi(img)
    crop = img[y:y + h, x:x + w]
    if crop.size == 0:
        crop = img
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_array(samples):
    X, keep = [], []
    for i, s in enumerate(samples):
        arr = load_roi(s["path"])
        if arr is None:
            continue
        X.append(arr)
        keep.append(s)
        if (i + 1) % 100 == 0:
            print(f"    loaded {i + 1}/{len(samples)}")
    return np.asarray(X, dtype="uint8"), keep


# --------------------------------------------------------------------------- #
# Metrics (same schema as train_tier2_svm.py)
# --------------------------------------------------------------------------- #
def evaluate(y, p, thr):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, confusion_matrix,
                                 average_precision_score, roc_auc_score)
    yp = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)
    return dict(threshold=round(float(thr), 3), accuracy=round(accuracy_score(y, yp), 4),
                balanced_accuracy=round(balanced_accuracy_score(y, yp), 4),
                pr_auc_adulterated=round(average_precision_score(y, p), 4),
                roc_auc=round(roc_auc_score(y, p), 4),
                pure_precision=round(float(pr[0]), 4), pure_recall=round(float(rc[0]), 4),
                pure_f1=round(float(f1[0]), 4),
                adul_precision=round(float(pr[1]), 4), adul_recall=round(float(rc[1]), 4),
                adul_f1=round(float(f1[1]), 4), macro_f1=round(float(f1.mean()), 4),
                confusion=confusion_matrix(y, yp, labels=[0, 1]).tolist())


def best_threshold(y, p):
    from sklearn.metrics import balanced_accuracy_score
    ts = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    best, bt = -1, 0.5
    for t in ts:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best:
            best, bt = ba, t
    return float(bt)


# --------------------------------------------------------------------------- #
# Model / data (PyTorch)
# --------------------------------------------------------------------------- #
def build_model(fine_tune):
    import torch.nn as nn
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    if fine_tune:  # unfreeze only the last feature block to avoid overfitting a small set
        for p in model.features[-1].parameters():
            p.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, 1)  # single logit -> BCEWithLogits
    return model


def make_loader(X, y, train, batch):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    tf_eval = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    tf_train = transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    class ArrDS(Dataset):
        def __init__(self, X, y, tf):
            self.X, self.y, self.tf = X, y, tf

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.tf(self.X[i]), torch.tensor(float(self.y[i]))

    ds = ArrDS(X, y, tf_train if train else tf_eval)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=0)


def predict_probs(model, X, device, batch=32):
    import torch
    loader = make_loader(X, np.zeros(len(X)), train=False, batch=batch)
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device)).ravel()
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def train_variant(Xtr, ytr, Xva, yva, args, use_weight, device):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(args.fine_tune).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # sklearn 'balanced' weights: w[c] = n_total / (2 * n_c)
    cw = None
    if use_weight:
        n0, n1 = (ytr == 0).sum(), (ytr == 1).sum()
        tot = n0 + n1
        cw = {0: tot / (2 * n0), 1: tot / (2 * n1)}

    train_loader = make_loader(Xtr, ytr, train=True, batch=args.batch)
    val_loader = make_loader(Xva, yva, train=False, batch=args.batch)

    best_val, best_state, patience, bad = float("inf"), None, 5, 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).ravel()
            w = None
            if cw is not None:
                w = torch.where(yb > 0.5, torch.tensor(cw[1], device=device),
                                torch.tensor(cw[0], device=device))
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()

        # validation loss (unweighted, for early stopping)
        model.eval(); tot_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                tot_loss += F.binary_cross_entropy_with_logits(model(xb).ravel(), yb).item() * len(yb)
                n += len(yb)
        val_loss = tot_loss / max(n, 1)
        print(f"    epoch {epoch + 1}/{args.epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                print("    early stopping"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=str(ROOT / "data" / "split"))
    ap.add_argument("--frames-root", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--metadata", default=str(ROOT / "data" / "metadata.csv"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fine-tune", action="store_true",
                    help="Unfreeze the top MobileNetV2 block (else train head only).")
    ap.add_argument("--frames", dest="frames", action="store_true", default=True)
    ap.add_argument("--no-frames", dest="frames", action="store_false")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    EXP.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata, dtype=str).fillna("").set_index("file_name")
    stem2pile = build_stem_to_pile(meta)
    split_root, frames_root = Path(args.split_root), Path(args.frames_root)

    print("Enumerating samples...")
    tr_s = enumerate_split("train", meta, stem2pile, split_root, frames_root, args.frames)
    va_s = enumerate_split("val", meta, stem2pile, split_root, frames_root, args.frames)
    te_s = enumerate_split("test", meta, stem2pile, split_root, frames_root, args.frames)
    print(f"  train {len(tr_s)} | val {len(va_s)} | test {len(te_s)} "
          f"(frames {'ON' if args.frames else 'OFF'})")

    print("Loading + cropping ROIs (train)..."); Xtr, tr_s = load_array(tr_s)
    print("Loading + cropping ROIs (val)...");   Xva, va_s = load_array(va_s)
    print("Loading + cropping ROIs (test)...");  Xte, te_s = load_array(te_s)
    ytr = np.array([s["cls"] for s in tr_s]); yva = np.array([s["cls"] for s in va_s])
    yte = np.array([s["cls"] for s in te_s])
    print(f"  arrays: train {Xtr.shape} (pure {sum(ytr==0)}/adul {sum(ytr==1)}) | "
          f"val {Xva.shape} | test {Xte.shape} (pure {sum(yte==0)}/adul {sum(yte==1)})")

    results = {}
    for name, use_weight in [("unweighted", False), ("balanced", True)]:
        print(f"\n=== training variant: {name} ===")
        model = train_variant(Xtr, ytr, Xva, yva, args, use_weight, device)
        torch.save(model.state_dict(), EXP / f"model_{name}.pt")
        pva = predict_probs(model, Xva, device, args.batch)
        pte = predict_probs(model, Xte, device, args.batch)
        thr = best_threshold(yva, pva)
        results[name] = {"test_default_0.5": evaluate(yte, pte, 0.5),
                         "test_val_threshold": evaluate(yte, pte, thr),
                         "val_selected_threshold": thr}
        if name == "balanced":
            pd.DataFrame([dict(file_name=s["file_name"], rel_path=s["path"],
                               pile_id=s["pile_id"], ratio=s["ratio"], cls=s["cls"],
                               source=s["source"], score=float(pte[i]))
                          for i, s in enumerate(te_s)]).to_csv(EXP / "test_scores.csv", index=False)
            pd.DataFrame([dict(file_name=s["file_name"], pile_id=s["pile_id"], ratio=s["ratio"],
                               cls=s["cls"], score=float(pva[i]))
                          for i, s in enumerate(va_s)]).to_csv(EXP / "val_scores.csv", index=False)
            _pte_balanced = pte

    json.dump({"model": f"MobileNetV2 transfer (fine_tune={args.fine_tune}) on 224x224 ROI crops [PyTorch]",
               "seed": SEED, "used_frames": args.frames,
               "test_class_counts": {"pure": int(sum(yte == 0)), "adulterated": int(sum(yte == 1))},
               "results": results}, open(EXP / "metrics.json", "w"), indent=2)

    # ---- comparison across all three tiers ----
    try:
        t1 = json.load(open(ROOT / "experiments/exp_001_logreg/metrics.json"))["results"]
        t2 = json.load(open(ROOT / "experiments/exp_002_svm/metrics.json"))["results"]
        lines = ["# Tier-1 (LogReg) vs Tier-2 (SVM hist) vs Tier-3 (MobileNetV2)\n",
                 "Sealed test. Positive = adulterated.\n",
                 "| model | variant | threshold | bal_acc | PR-AUC | pure_recall | adul_recall | accuracy |",
                 "|---|---|---|---|---|---|---|---|"]

        def grab(res, var, pt):
            r = res[var][pt]
            return (r["balanced_accuracy"], r["pr_auc_adulterated"], r["pure_recall"],
                    r["adul_recall"], r["accuracy"])

        for tag, res in [("Tier-1 LogReg", t1), ("Tier-2 SVM", t2), ("Tier-3 CNN", results)]:
            for var in ["unweighted", "balanced"]:
                for pt, lab in [("test_default_0.5", "0.5"), ("test_val_threshold", "val-thr")]:
                    b, pa, pu, ad, ac = grab(res, var, pt)
                    lines.append(f"| {tag} | {var} | {lab} | {b:.3f} | {pa:.3f} | {pu:.2f} | {ad:.2f} | {ac:.3f} |")
        (EXP / "comparison_all_tiers.md").write_text("\n".join(lines) + "\n")
    except FileNotFoundError as e:
        print(f"  (comparison skipped: {e})")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    prec, rec, _ = precision_recall_curve(yte, _pte_balanced)
    ax[0].plot(rec, prec, color="#8e44ad",
               label=f"CNN balanced (AP={average_precision_score(yte, _pte_balanced):.3f})")
    ax[0].axhline(sum(yte == 1) / len(yte), ls="--", color="gray",
                  label=f"prevalence={sum(yte==1)/len(yte):.2f}")
    ax[0].set_xlabel("recall (adulterated)"); ax[0].set_ylabel("precision")
    ax[0].set_title("Tier-3 CNN — test PR"); ax[0].legend()
    names = ["unweighted", "balanced"]
    ba = [results[n]["test_val_threshold"]["balanced_accuracy"] for n in names]
    prc = [results[n]["test_val_threshold"]["pure_recall"] for n in names]
    arc = [results[n]["test_val_threshold"]["adul_recall"] for n in names]
    x = np.arange(len(names)); w = .25
    ax[1].bar(x - w, ba, w, label="balanced acc", color="#27ae60")
    ax[1].bar(x, prc, w, label="pure recall", color="#e67e22")
    ax[1].bar(x + w, arc, w, label="adul recall", color="#6b4f2a")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Tier-3 test @ val threshold"); ax[1].legend()
    plt.tight_layout(); plt.savefig(EXP / "pr_and_metrics.png", dpi=150); plt.close()

    print(f"\nSaved to {EXP}")
    for n in names:
        r = results[n]["test_val_threshold"]
        print(f"  {n:<11} bal_acc={r['balanced_accuracy']:.3f} PR-AUC={r['pr_auc_adulterated']:.3f} "
              f"ROC-AUC={r['roc_auc']:.3f} pure_R={r['pure_recall']:.2f} adul_R={r['adul_recall']:.2f}")


if __name__ == "__main__":
    main()
