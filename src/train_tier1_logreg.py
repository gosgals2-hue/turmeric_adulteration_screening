"""
train_tier1_logreg.py — Tier-1 baseline: logistic regression on masked colour features.

Trains on the TRAIN split, selects the decision threshold on VAL, and reports on the
SEALED TEST split. Runs two variants — unweighted and class_weight='balanced' — and
writes everything to experiments/exp_001_logreg/.

Positive class = adulterated (cls 1). Because the test set is imbalanced (~12 pure :
103 adulterated), a trivial "always adulterated" classifier scores ~90% accuracy but
50% balanced accuracy and 0% pure recall — so we report accuracy, BALANCED accuracy,
per-class precision/recall/F1, PR-AUC (average precision), and the confusion matrix.

Run from project root:  python src/train_tier1_logreg.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, average_precision_score, precision_recall_curve)

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT/"data"/"metadata.csv").exists(): break
    ROOT = ROOT.parent
EXP = ROOT/"experiments"/"exp_001_logreg"; EXP.mkdir(parents=True, exist_ok=True)
FEATURES = ["mean_R","mean_G","mean_B","mean_H","mean_S","mean_V",
            "hue_std","sat_std","val_std","bright","contrast"]
SEED = 42

tv = pd.read_csv(ROOT/"eda/features_trainval.csv")
te = pd.read_csv(ROOT/"eda/features_test.csv")
tr = tv[tv["split"]=="train"].copy(); va = tv[tv["split"]=="val"].copy()
Xtr, ytr = tr[FEATURES].values, tr["cls"].values
Xva, yva = va[FEATURES].values, va["cls"].values
Xte, yte = te[FEATURES].values, te["cls"].values

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
print(f"train n={len(ytr)} (pure {sum(ytr==0)}/adul {sum(ytr==1)}) | "
      f"val n={len(yva)} | test n={len(yte)} (pure {sum(yte==0)}/adul {sum(yte==1)})")

def best_threshold(y, p):
    # threshold on val maximizing balanced accuracy (positive=adulterated)
    ts = np.unique(np.concatenate([[0.0], np.sort(p), [1.0]]))
    best, bt = -1, 0.5
    for t in ts:
        ba = balanced_accuracy_score(y, (p>=t).astype(int))
        if ba > best: best, bt = ba, t
    return float(bt)

def evaluate(y, p, thr):
    yp = (p>=thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, labels=[0,1], zero_division=0)
    return dict(
        threshold=round(float(thr),3),
        accuracy=round(accuracy_score(y,yp),4),
        balanced_accuracy=round(balanced_accuracy_score(y,yp),4),
        pr_auc_adulterated=round(average_precision_score(y,p),4),
        pure_precision=round(float(pr[0]),4), pure_recall=round(float(rc[0]),4), pure_f1=round(float(f1[0]),4),
        adul_precision=round(float(pr[1]),4), adul_recall=round(float(rc[1]),4), adul_f1=round(float(f1[1]),4),
        macro_f1=round(float(f1.mean()),4),
        confusion=confusion_matrix(y,yp,labels=[0,1]).tolist())

results = {}
models = {}
for name, kw in [("unweighted", {}), ("balanced", {"class_weight":"balanced"})]:
    clf = LogisticRegression(max_iter=2000, random_state=SEED, **kw).fit(Xtr_s, ytr)
    models[name] = clf
    pva = clf.predict_proba(Xva_s)[:,1]; pte = clf.predict_proba(Xte_s)[:,1]
    thr = best_threshold(yva, pva)
    results[name] = {
        "test_default_0.5": evaluate(yte, pte, 0.5),
        "test_val_threshold": evaluate(yte, pte, thr),
        "val_selected_threshold": thr,
        "coefficients": dict(zip(FEATURES, np.round(clf.coef_[0],4).tolist())),
        "intercept": round(float(clf.intercept_[0]),4),
    }

json.dump({"features":FEATURES,"seed":SEED,
           "train_n":int(len(ytr)),"val_n":int(len(yva)),"test_n":int(len(yte)),
           "test_class_counts":{"pure":int(sum(yte==0)),"adulterated":int(sum(yte==1))},
           "note":"Image-only features (no video frames). Positive class=adulterated.",
           "results":results}, open(EXP/"metrics.json","w"), indent=2)

# ---- figures ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
for name,c in [("unweighted","#2980b9"),("balanced","#c0392b")]:
    pte=models[name].predict_proba(Xte_s)[:,1]
    prec,rec,_=precision_recall_curve(yte,pte); ap=average_precision_score(yte,pte)
    ax[0].plot(rec,prec,color=c,label=f"{name} (AP={ap:.3f})")
base=sum(yte==1)/len(yte)
ax[0].axhline(base,ls="--",color="gray",label=f"baseline (prevalence={base:.2f})")
ax[0].set_xlabel("recall (adulterated)"); ax[0].set_ylabel("precision"); ax[0].set_title("Test PR curve"); ax[0].legend()
# balanced accuracy bar comparison at val threshold
names=["unweighted","balanced"]
ba=[results[n]["test_val_threshold"]["balanced_accuracy"] for n in names]
acc=[results[n]["test_val_threshold"]["accuracy"] for n in names]
pr=[results[n]["test_val_threshold"]["pure_recall"] for n in names]
x=np.arange(len(names)); w=.25
ax[1].bar(x-w,acc,w,label="accuracy",color="#7f8c8d")
ax[1].bar(x,ba,w,label="balanced acc",color="#27ae60")
ax[1].bar(x+w,pr,w,label="pure recall",color="#e67e22")
ax[1].axhline(0.5,ls=":",color="gray"); ax[1].set_xticks(x); ax[1].set_xticklabels(names)
ax[1].set_ylim(0,1.05); ax[1].set_title("Test metrics @ val-selected threshold"); ax[1].legend()
plt.tight_layout(); plt.savefig(EXP/"pr_and_metrics.png",dpi=150); plt.close()

# confusion matrices
fig,ax=plt.subplots(1,2,figsize=(9,4))
for k,name in enumerate(names):
    cm=np.array(results[name]["test_val_threshold"]["confusion"])
    ax[k].imshow(cm,cmap="Blues"); ax[k].set_title(f"{name}\n(val-thr {results[name]['val_selected_threshold']:.2f})")
    for (i,j),val in np.ndenumerate(cm): ax[k].text(j,i,val,ha="center",va="center",fontsize=13)
    ax[k].set_xticks([0,1]); ax[k].set_xticklabels(["pure","adul"]); ax[k].set_yticks([0,1]); ax[k].set_yticklabels(["pure","adul"])
    ax[k].set_xlabel("predicted"); ax[k].set_ylabel("true")
plt.tight_layout(); plt.savefig(EXP/"confusion.png",dpi=150); plt.close()

# console summary
def row(n,r): return (f"  {n:<11} acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
    f"PR-AUC={r['pr_auc_adulterated']:.3f} | pure P/R/F1={r['pure_precision']:.2f}/{r['pure_recall']:.2f}/{r['pure_f1']:.2f} "
    f"| adul R={r['adul_recall']:.2f}")
print("\n=== TEST @ default 0.5 ===")
for n in names: print(row(n,results[n]["test_default_0.5"]))
print("=== TEST @ val-selected threshold ===")
for n in names: print(row(n,results[n]["test_val_threshold"]))
print(f"\nSaved to {EXP}")
