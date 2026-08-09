"""
evaluate_protocol.py — Approved evaluation protocol (see PROTOCOL_evaluation.md), for ANY tier.

On the leak-free TEST pool, for a chosen model, produces:
  A. Per-ratio sensitivity (limit-of-detection) with Wilson 95% CIs.
  B. Specificity on pure (with n-pure-pile caveat).
  C. Se@Sp operating points {0.90,0.95,0.98} — threshold chosen on VAL to hit target
     specificity, then test sensitivity reported.
  D. Deployment metrics across a PREVALENCE SWEEP {99:1, 95:5, 90:10} computed analytically.
  E. Overall-Se CI via GROUP bootstrap over adulterated test PILES (not frames).

Runs on Tier-1 (LogReg, 11 colour stats), Tier-2 (SVM, 96-d histograms; uses the tuned
hyperparameters train_tier2_svm.py saved), and Tier-3 (CNN; reads its saved test/val scores).
Each model's outputs go to its own experiments/<exp>/protocol_eval/ folder.

Run from project root:
    python src/evaluate_protocol.py                      # all available tiers, balanced
    python src/evaluate_protocol.py --model tier2
    python src/evaluate_protocol.py --model tier1 tier2  # a subset
    python src/evaluate_protocol.py --class-weight none
"""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

warnings.filterwarnings("ignore")  # silence sklearn SVC(probability=...) deprecation spam

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT/"data"/"metadata.csv").exists(): break
    ROOT = ROOT.parent
FEATURES = ["mean_R","mean_G","mean_B","mean_H","mean_S","mean_V","hue_std","sat_std","val_std","bright","contrast"]
PREVALENCES = [0.99, 0.95, 0.90]
TARGET_SP = [0.90, 0.95, 0.98]
EXP_DIR = {"tier1": "exp_001_logreg", "tier2": "exp_002_svm", "tier3": "exp_003_cnn"}
LABEL = {"tier1": "Tier-1 LogReg (11 colour stats)", "tier2": "Tier-2 SVM (96-d histograms)",
         "tier3": "Tier-3 MobileNetV2 (CNN)"}


def rkey(r):
    try: return -float(str(r).split("_")[0].replace("/", "_").split("/")[0])
    except: return 0
rlabel = lambda r: "pure" if str(r) in ("100_0","") else str(r).replace("_","/")


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    ph=k/n; d=1+z*z/n; c=(ph+z*z/(2*n))/d
    h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)


def thr_for_sp(pure_scores, target_sp):
    if len(pure_scores)==0: return 0.5
    return float(np.quantile(pure_scores, target_sp, method="higher"))


def deployment(se, sp, pi_pure):
    pi_a=1-pi_pure
    ppv=(pi_a*se)/(pi_a*se+pi_pure*(1-sp)) if (pi_a*se+pi_pure*(1-sp))>0 else float("nan")
    npv=(pi_pure*sp)/(pi_pure*sp+pi_a*(1-se)) if (pi_pure*sp+pi_a*(1-se))>0 else float("nan")
    alert=pi_a*se+pi_pure*(1-sp); wacc=pi_pure*sp+pi_a*se
    return dict(pi_pure=pi_pure, PPV=ppv, NPV=npv, alert_rate=alert, weighted_acc=wacc,
                per1000=dict(true_detections=round(1000*pi_a*se), missed=round(1000*pi_a*(1-se)),
                             false_alarms=round(1000*pi_pure*(1-sp)), true_negatives=round(1000*pi_pure*sp)))


# --------------------------------------------------------------------------- #
# Per-model scores: returns (pva, yva, pte, yte, te_df) with te_df index 0..N-1
# --------------------------------------------------------------------------- #
def scores_tier1(cw):
    tv=pd.read_csv(ROOT/"eda/features_trainval.csv"); te=pd.read_csv(ROOT/"eda/features_test.csv").reset_index(drop=True)
    tr=tv[tv.split=="train"]; va=tv[tv.split=="val"]
    sc=StandardScaler().fit(tr[FEATURES].values)
    clf=LogisticRegression(max_iter=2000,random_state=42,class_weight=cw).fit(sc.transform(tr[FEATURES].values), tr.cls.values)
    pva=clf.predict_proba(sc.transform(va[FEATURES].values))[:,1]
    pte=clf.predict_proba(sc.transform(te[FEATURES].values))[:,1]
    return pva, va.cls.values, pte, te.cls.values, te[["cls","ratio","pile_id"]]


def scores_tier2(cw):
    df=pd.read_csv(ROOT/"eda/hist_features.csv")
    H=[c for c in df.columns if c.startswith("h") and c[1] in "RGB"]
    tr=df[df.split=="train"]; va=df[df.split=="val"]; te=df[df.split=="test"].reset_index(drop=True)
    cwkey="balanced" if cw=="balanced" else "unweighted"
    C, gamma = 10, "scale"
    try:
        bp=json.load(open(ROOT/"experiments/exp_002_svm/metrics.json"))["tuning"][cwkey]["best_params"]
        C, gamma = bp.get("C",10), bp.get("gamma","scale")
    except Exception: pass
    sc=StandardScaler().fit(tr[H].values)
    clf=SVC(kernel="rbf",C=C,gamma=gamma,probability=True,random_state=42,class_weight=cw).fit(sc.transform(tr[H].values), tr.cls.values)
    pva=clf.predict_proba(sc.transform(va[H].values))[:,1]
    pte=clf.predict_proba(sc.transform(te[H].values))[:,1]
    return pva, va.cls.values, pte, te.cls.values, te[["cls","ratio","pile_id"]]


def scores_tier3(cw):
    tsp=ROOT/"experiments/exp_003_cnn/test_scores.csv"; vsp=ROOT/"experiments/exp_003_cnn/val_scores.csv"
    if not (tsp.exists() and vsp.exists()):
        return None
    te=pd.read_csv(tsp).reset_index(drop=True); va=pd.read_csv(vsp)
    return va.score.values, va.cls.values, te.score.values, te.cls.values, te[["cls","ratio","pile_id"]]


SCORERS = {"tier1": scores_tier1, "tier2": scores_tier2, "tier3": scores_tier3}


def run_protocol(model, cw, cw_str, boot):
    got = SCORERS[model](cw)
    if got is None:
        print(f"[{model}] scores not found (run its trainer first) — skipping."); return
    pva, yva, pte, yte, te = got
    te = te.reset_index(drop=True); yte = np.asarray(yte)
    OUT = ROOT/"experiments"/EXP_DIR[model]/"protocol_eval"; OUT.mkdir(parents=True, exist_ok=True)
    pure_scores_val = pva[yva==0]
    roc_auc=roc_auc_score(yte,pte); ap_sc=average_precision_score(yte,pte)

    ops=[]
    for tsp in TARGET_SP:
        thr=thr_for_sp(pure_scores_val, tsp); yp=(pte>=thr).astype(int)
        pure=yte==0; adul=yte==1
        ops.append(dict(target_sp=tsp, threshold=round(thr,3),
                        test_specificity=round(float(np.mean(yp[pure]==0)),3) if pure.sum() else float("nan"),
                        test_sensitivity_overall=round(float(np.mean(yp[adul]==1)),3) if adul.sum() else float("nan")))
    pd.DataFrame(ops).to_csv(OUT/"operating_points.csv",index=False)
    prim=[o for o in ops if o["target_sp"]==0.95][0]
    thr=prim["threshold"]; se_overall=prim["test_sensitivity_overall"]; sp_overall=prim["test_specificity"]

    dye=te[te.cls==1]
    rows=[]
    for r in sorted(dye.ratio.unique(), key=rkey):
        sub=dye[dye.ratio==r]; k=int((pte[sub.index]>=thr).sum()); n=len(sub); lo,hi=wilson(k,n)
        rows.append(dict(ratio=rlabel(r), n=n, detected=k, sensitivity=round(k/n,3),
                         ci_lo=round(lo,3), ci_hi=round(hi,3), n_test_piles=int(sub.pile_id.nunique())))
    sens=pd.DataFrame(rows); sens.to_csv(OUT/"sensitivity_by_ratio.csv",index=False)

    pure=te[te.cls==0]; kp=int((pte[pure.index]<thr).sum()); npu=len(pure); sp_lo,sp_hi=wilson(kp,npu)
    npp=int(pure.pile_id.nunique())
    spec=dict(specificity=round(kp/npu,3), n=npu, ci_lo=round(sp_lo,3), ci_hi=round(sp_hi,3),
              n_pure_test_piles=npp,
              caveat=("specificity now rests on %d physical pure pile(s)" % npp) +
                     (" - including different-source probe piles" if npp>=2 else "; CI understates true uncertainty"))

    rng=np.random.default_rng(42); piles=dye.pile_id.unique()
    idx_by_pile={p:dye[dye.pile_id==p].index.values for p in piles}; ses=[]
    for _ in range(boot):
        samp=rng.choice(piles,size=len(piles),replace=True)
        ii=np.concatenate([idx_by_pile[p] for p in samp]); ses.append(np.mean(pte[ii]>=thr))
    se_ci=(round(float(np.percentile(ses,2.5)),3), round(float(np.percentile(ses,97.5)),3))
    sweep=[deployment(se_overall, sp_overall, pp) for pp in PREVALENCES]

    fig,ax=plt.subplots(1,2,figsize=(14,5))
    xr=np.arange(len(sens))
    ax[0].errorbar(xr, sens.sensitivity, yerr=[sens.sensitivity-sens.ci_lo, sens.ci_hi-sens.sensitivity],
                   marker="o", capsize=4, color="#6b4f2a")
    ax[0].axhline(0.9, ls="--", color="green", label="0.90 reliability line")
    ax[0].set_xticks(xr); ax[0].set_xticklabels(sens.ratio, rotation=45, ha="right")
    ax[0].set_ylim(0,1.05); ax[0].set_ylabel("sensitivity (adulterated recall)")
    ax[0].set_title(f"{LABEL[model]} — per-ratio sensitivity @ Sp≈{sp_overall:.2f} (thr={thr:.2f})"); ax[0].legend()
    fpr,tpr,_=roc_curve(yte,pte)
    ax[1].plot(fpr,tpr,color="#2980b9",label=f"ROC (AUC={roc_auc:.3f})"); ax[1].plot([0,1],[0,1],ls=":",color="gray")
    ax[1].set_xlabel("false-positive rate (1−Sp)"); ax[1].set_ylabel("sensitivity"); ax[1].set_title("Test ROC"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT/"sensitivity_and_roc.png",dpi=150); plt.close()

    md=[f"# Deployment metrics — prevalence sweep ({LABEL[model]})\n",
        f"class_weight={cw_str}. Threshold chosen on VAL for target specificity 0.95 -> "
        f"test Sp={sp_overall:.2f}, overall test Se={se_overall:.2f} (pile-bootstrap 95% CI "
        f"{se_ci[0]}–{se_ci[1]}). ROC-AUC={roc_auc:.3f}, PR-AUC={ap_sc:.3f}. Specificity rests on "
        f"{npp} pure pile(s).\n",
        "| scenario (pure:adul) | PPV | NPV | alert rate | weighted acc | per-1000: detected / missed / false-alarms |",
        "|---|---|---|---|---|---|"]
    for pp,s in zip(PREVALENCES,sweep):
        p=s["per1000"]
        md.append(f"| {int(pp*100)}:{int((1-pp)*100)} | {s['PPV']:.2f} | {s['NPV']:.3f} | {s['alert_rate']:.3f} | "
                  f"{s['weighted_acc']:.3f} | {p['true_detections']} / {p['missed']} / {p['false_alarms']} |")
    md.append("\n## Se@Sp operating points (threshold from val)\n| target Sp | threshold | test Sp | overall test Se |\n|---|---|---|---|")
    for o in ops: md.append(f"| {o['target_sp']} | {o['threshold']} | {o['test_specificity']} | {o['test_sensitivity_overall']} |")
    (OUT/"deployment_metrics.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    json.dump(dict(model=model, class_weight=cw_str, roc_auc=round(roc_auc,4), pr_auc=round(ap_sc,4),
                   primary_threshold=thr, se_overall=se_overall, se_overall_pile_ci=se_ci,
                   specificity=spec, operating_points=ops, prevalence_sweep=sweep,
                   sensitivity_by_ratio=rows), open(OUT/"protocol_metrics.json","w"), indent=2)

    print(f"\n===== {LABEL[model]} =====")
    print("per-ratio sensitivity @ Sp~%.2f (thr=%.2f):"%(sp_overall,thr))
    print(sens.to_string(index=False))
    print(f"specificity(pure) = {spec['specificity']} (n={npu}, {npp} piles, CI {spec['ci_lo']}-{spec['ci_hi']}) | "
          f"overall Se {se_overall} pile-CI {se_ci} | ROC-AUC {roc_auc:.3f} PR-AUC {ap_sc:.3f}")
    print("deployment sweep (PPV / alert / false-alarms per 1000):")
    for pp,s in zip(PREVALENCES,sweep):
        print(f"  pure {int(pp*100)}%: PPV={s['PPV']:.2f} alert={s['alert_rate']:.3f} FA/1000={s['per1000']['false_alarms']}")
    print(f"Saved to {OUT}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["all"], choices=["tier1","tier2","tier3","all"])
    ap.add_argument("--class-weight", default="balanced", choices=["balanced","none"])
    ap.add_argument("--boot", type=int, default=2000)
    args=ap.parse_args()
    cw = None if args.class_weight=="none" else "balanced"
    models = ["tier1","tier2","tier3"] if "all" in args.model else args.model
    for m in models:
        run_protocol(m, cw, args.class_weight, args.boot)


if __name__=="__main__":
    main()
