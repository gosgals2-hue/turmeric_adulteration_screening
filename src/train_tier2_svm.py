"""
train_tier2_svm.py — Tier-2 baseline: RBF SVM on 32-bin RGB colour histograms.

Same protocol as Tier-1: train on TRAIN, threshold on VAL, report on sealed TEST;
unweighted + class_weight='balanced'; positive class = adulterated. Writes to
experiments/exp_002_svm/ and prints a Tier-1 vs Tier-2 comparison.

Hyperparameters (C, gamma) are chosen by a grid search that is itself pile-aware:
GridSearchCV with GroupKFold over pile_id, so no physical pile straddles a CV fold.
This closes the "the split was leak-free but the tuning wasn't" gap — both the final
split and the hyperparameter search respect pile boundaries. Default CV metric is
balanced_accuracy (robust when a fold happens to be single-class on this small set);
use --scoring roc_auc once more pure piles make every fold two-class. --no-tune
reverts to the previous fixed C=10, gamma=scale.

Run from project root:  python src/train_tier2_svm.py            # pile-aware tuned
                        python src/train_tier2_svm.py --no-tune  # fixed C=10 gamma=scale
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, average_precision_score, precision_recall_curve, roc_auc_score)

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT/"data"/"metadata.csv").exists(): break
    ROOT = ROOT.parent
EXP = ROOT/"experiments"/"exp_002_svm"; EXP.mkdir(parents=True, exist_ok=True)
SEED = 42

ap = argparse.ArgumentParser(description="Tier-2 SVM with pile-aware (GroupKFold) C/gamma search.")
ap.add_argument("--no-tune", action="store_true", help="Skip the search; use fixed C=10 gamma=scale.")
ap.add_argument("--scoring", default="balanced_accuracy",
                help="CV scoring metric for the search (balanced_accuracy [default, robust to "
                     "single-class folds], roc_auc, f1_macro, ...).")
ARGS = ap.parse_args()
# C/gamma grid searched with GroupKFold over pile_id so no pile straddles a CV fold — the same
# leak-avoidance the final train/test split uses, applied to hyperparameter selection.
PARAM_GRID = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1, 1]}

df = pd.read_csv(ROOT/"eda/hist_features.csv")
HFEATS = [c for c in df.columns if c.startswith("h") and c[1] in "RGB"]
tr, va, te = df[df.split=="train"], df[df.split=="val"], df[df.split=="test"]
Xtr,ytr = tr[HFEATS].values, tr.cls.values
Xva,yva = va[HFEATS].values, va.cls.values
Xte,yte = te[HFEATS].values, te.cls.values
sc = StandardScaler().fit(Xtr)
Xtr_s,Xva_s,Xte_s = sc.transform(Xtr),sc.transform(Xva),sc.transform(Xte)
print(f"features={len(HFEATS)} | train n={len(ytr)} (pure {sum(ytr==0)}/adul {sum(ytr==1)}) | "
      f"val n={len(yva)} | test n={len(yte)} (pure {sum(yte==0)}/adul {sum(yte==1)})")

def best_threshold(y,p):
    ts=np.unique(np.concatenate([[0.0],np.sort(p),[1.0]])); best,bt=-1,0.5
    for t in ts:
        ba=balanced_accuracy_score(y,(p>=t).astype(int))
        if ba>best: best,bt=ba,t
    return float(bt)

def evaluate(y,p,thr):
    yp=(p>=thr).astype(int)
    pr,rc,f1,_=precision_recall_fscore_support(y,yp,labels=[0,1],zero_division=0)
    return dict(threshold=round(float(thr),3),accuracy=round(accuracy_score(y,yp),4),
        balanced_accuracy=round(balanced_accuracy_score(y,yp),4),
        pr_auc_adulterated=round(average_precision_score(y,p),4),
        roc_auc=round(roc_auc_score(y,p),4),
        pure_precision=round(float(pr[0]),4),pure_recall=round(float(rc[0]),4),pure_f1=round(float(f1[0]),4),
        adul_precision=round(float(pr[1]),4),adul_recall=round(float(rc[1]),4),adul_f1=round(float(f1[1]),4),
        macro_f1=round(float(f1.mean()),4),confusion=confusion_matrix(y,yp,labels=[0,1]).tolist())

groups_tr = tr["pile_id"].values          # pile of each TRAIN sample, for GroupKFold
n_groups = int(tr["pile_id"].nunique())
n_splits = max(2, min(5, n_groups))
can_tune = (not ARGS.no_tune) and n_groups >= 2
if not ARGS.no_tune and n_groups < 2:
    print(f"  WARN: only {n_groups} train pile(s) — cannot GroupKFold; using fixed C=10 gamma=scale.")
print(f"tuning: {'GroupKFold(pile_id) n_splits=%d over %d piles, scoring=%s' % (n_splits, n_groups, ARGS.scoring) if can_tune else 'OFF (fixed C=10, gamma=scale)'}")

results={}; models={}; tuning={}
for name,kw in [("unweighted",{}),("balanced",{"class_weight":"balanced"})]:
    if can_tune:
        gs=GridSearchCV(SVC(kernel="rbf",probability=True,random_state=SEED,**kw),
                        PARAM_GRID, scoring=ARGS.scoring, cv=GroupKFold(n_splits=n_splits),
                        n_jobs=-1, refit=True)
        gs.fit(Xtr_s, ytr, groups=groups_tr)
        clf=gs.best_estimator_
        tuning[name]={"cv":"GroupKFold(groups=pile_id)","n_splits":n_splits,"n_train_piles":n_groups,
                      "scoring":ARGS.scoring,"best_params":gs.best_params_,
                      "cv_score":round(float(gs.best_score_),4)}
        print(f"  [{name}] best {gs.best_params_}  cv {ARGS.scoring}={gs.best_score_:.3f}")
    else:
        clf=SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=SEED,**kw).fit(Xtr_s,ytr)
        tuning[name]={"best_params":{"C":10,"gamma":"scale"},"note":"fixed (tuning skipped)"}
    models[name]=clf
    pva=clf.predict_proba(Xva_s)[:,1]; pte=clf.predict_proba(Xte_s)[:,1]
    thr=best_threshold(yva,pva)
    results[name]={"test_default_0.5":evaluate(yte,pte,0.5),"test_val_threshold":evaluate(yte,pte,thr),
                   "val_selected_threshold":thr,
                   "hyperparams":{"C":clf.C,"gamma":clf.gamma}}

json.dump({"model":"SVC rbf on 32-bin RGB histograms (masked); C/gamma via GroupKFold(pile_id) unless --no-tune",
           "features":len(HFEATS),"seed":SEED,"tuning":tuning,
           "test_class_counts":{"pure":int(sum(yte==0)),"adulterated":int(sum(yte==1))},
           "results":results},open(EXP/"metrics.json","w"),indent=2)

# PR figure vs Tier-1
fig,ax=plt.subplots(1,2,figsize=(13,5))
for name,c in [("unweighted","#2980b9"),("balanced","#c0392b")]:
    pte=models[name].predict_proba(Xte_s)[:,1]; prec,rec,_=precision_recall_curve(yte,pte); apx=average_precision_score(yte,pte)
    ax[0].plot(rec,prec,color=c,label=f"SVM {name} (AP={apx:.3f})")
ax[0].axhline(sum(yte==1)/len(yte),ls="--",color="gray",label=f"prevalence={sum(yte==1)/len(yte):.2f}")
ax[0].set_xlabel("recall (adulterated)"); ax[0].set_ylabel("precision"); ax[0].set_title("Tier-2 SVM — test PR"); ax[0].legend()
names=["unweighted","balanced"]
ba=[results[n]["test_val_threshold"]["balanced_accuracy"] for n in names]
pr=[results[n]["test_val_threshold"]["pure_recall"] for n in names]
ar=[results[n]["test_val_threshold"]["adul_recall"] for n in names]
x=np.arange(len(names)); w=.25
ax[1].bar(x-w,ba,w,label="balanced acc",color="#27ae60"); ax[1].bar(x,pr,w,label="pure recall",color="#e67e22")
ax[1].bar(x+w,ar,w,label="adul recall",color="#6b4f2a"); ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylim(0,1.05)
ax[1].axhline(0.5,ls=":",color="gray"); ax[1].set_title("Tier-2 test @ val threshold"); ax[1].legend()
plt.tight_layout(); plt.savefig(EXP/"pr_and_metrics.png",dpi=150); plt.close()

# ---- Tier-1 vs Tier-2 comparison ----
t1=json.load(open(ROOT/"experiments/exp_001_logreg/metrics.json"))["results"]
def grab(res,var,pt): r=res[var][pt]; return (r["balanced_accuracy"],r["pr_auc_adulterated"],r["pure_recall"],r["adul_recall"],r["accuracy"])
lines=["# Tier-1 (LogReg, 11 colour stats) vs Tier-2 (SVM, 96-d RGB histograms)\n",
 "Image-only, sealed test (12 pure : 103 adulterated). Positive = adulterated. PROVISIONAL (pre-frames).\n",
 "| model | variant | threshold | bal_acc | PR-AUC | pure_recall | adul_recall | accuracy |","|---|---|---|---|---|---|---|---|"]
for tag,res in [("Tier-1 LogReg",t1),("Tier-2 SVM",results)]:
    for var in ["unweighted","balanced"]:
        for pt,lab in [("test_default_0.5","0.5"),("test_val_threshold","val-thr")]:
            b,pa,pu,ad,ac=grab(res,var,pt)
            lines.append(f"| {tag} | {var} | {lab} | {b:.3f} | {pa:.3f} | {pu:.2f} | {ad:.2f} | {ac:.3f} |")
(EXP/"comparison_tier1_vs_tier2.md").write_text("\n".join(lines)+"\n")

def row(n,r): return (f"  {n:<11} acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
    f"PR-AUC={r['pr_auc_adulterated']:.3f} ROC-AUC={r['roc_auc']:.3f} | pure_R={r['pure_recall']:.2f} adul_R={r['adul_recall']:.2f}")
print("\n=== TIER-2 SVM  TEST @ default 0.5 ===")
for n in names: print(row(n,results[n]["test_default_0.5"]))
print("=== TIER-2 SVM  TEST @ val threshold ===")
for n in names: print(row(n,results[n]["test_val_threshold"]))
print(f"\nSaved to {EXP}")
