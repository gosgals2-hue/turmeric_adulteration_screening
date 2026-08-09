"""
eda_figures.py — Generate all EDA figures + a stats.json for captions.

Reads (no heavy image processing):
    data/metadata.csv                              (raw counts)
    data/split/manifest.csv                        (split assignment)
    eda/features_trainval.csv                      (masked color features, train+val)
    segmentation_review_v2/confidence_ranking.csv  (segmentation diagnostics)
    eda/train_video_frames.csv                     (projected frame counts)

Writes eda/figures/fig1..fig6 .png (150 dpi) and eda/stats.json.

Design note on lighting confound: raw mean RGB is dominated by the SVI screen
color, not the powder. So the color-vs-ratio trend (Fig 3) and the class-
separation PCA (Fig 4B) are computed on the AMBIENT subset only, where color
reflects the actual sample. The colored-light shots are used in Fig 5 to show
the illumination protocol expanded feature diversity. TEST is never used here.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists(): break
    ROOT = ROOT.parent
FIG = ROOT / "eda" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(ROOT / "data/metadata.csv", dtype=str).fillna("")
man = pd.read_csv(ROOT / "data/split/manifest.csv", dtype=str).fillna("")
feat = pd.read_csv(ROOT / "eda/features_trainval.csv")            # final: images + frames
te_feat = pd.read_csv(ROOT / "eda/features_test.csv")             # final test: images + frames
try:
    rank = pd.read_csv(ROOT / "segmentation_review_v2/confidence_ranking.csv")
except Exception:
    rank = None   # segmentation-QA figure is optional (one-time review of the initial set)

IMG_EXT = (".jpg", ".jpeg", ".png")
VID_EXT = (".mov", ".mp4", ".avi")

def is_img(s): return s.lower().endswith(IMG_EXT)
def is_vid(s): return s.lower().endswith(VID_EXT)

# ratio ordering by turmeric fraction descending (pure -> 50/50)
def rkey(r):
    try: return -float(r.split("_")[0])
    except: return 0
RATIOS = sorted(meta["ratio"].unique(), key=rkey)
def rlabel(r): return "pure" if r == "100_0" else r.replace("_", "/")

FEATURES = ["mean_R","mean_G","mean_B","mean_H","mean_S","mean_V",
            "hue_std","sat_std","val_std","bright","contrast"]

stats = {}

# ---------------------------------------------------------------------------
# FIG 1 — dataset summary
# ---------------------------------------------------------------------------
meta["is_img"] = meta["file_name"].map(is_img)
meta["is_vid"] = meta["file_name"].map(is_vid)
meta["type"] = meta["pile_id"].map(lambda p: "pure" if p.startswith("pure")
                                   else "yellow_dye" if p.startswith("yellow_dye") else "starch")

# Active dataset only (pure + yellow_dye). Starch shares ratios with dye, so
# grouping by ratio alone would pool them; starch is set aside and summarized
# separately below to avoid a misleading count.
active = meta[meta["type"].isin(["pure", "yellow_dye"])]
summ = []
for r in RATIOS:
    sub = active[active["ratio"] == r]
    if sub.empty: continue
    summ.append(dict(ratio=rlabel(r), type=sub["type"].iloc[0],
                     images=int(sub["is_img"].sum()), videos=int(sub["is_vid"].sum()),
                     lightings=sub["illumination_color"].nunique()))
summ_df = pd.DataFrame(summ)
starch = meta[meta["type"] == "starch"]
stats["starch_reserve"] = dict(images=int(starch["is_img"].sum()),
                               videos=int(starch["is_vid"].sum()),
                               ratios=int(starch["ratio"].nunique()))
summ_df.to_csv(ROOT / "eda/dataset_summary.csv", index=False)
stats["dataset_summary"] = summ_df.to_dict("records")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(summ_df))
axes[0].bar(x, summ_df["images"], label="images", color="#d98a29")
axes[0].bar(x, summ_df["videos"], bottom=summ_df["images"], label="videos", color="#6b4f2a")
axes[0].set_xticks(x); axes[0].set_xticklabels(summ_df["ratio"], rotation=45, ha="right")
axes[0].set_ylabel("file count"); axes[0].set_title("Media per ratio (all data)")
axes[0].legend()
# lighting distribution (images only)
lc = meta[meta["is_img"]]["illumination_color"].value_counts()
axes[1].bar(lc.index, lc.values, color="#c07b3a")
axes[1].set_title("Images per illumination condition (all)")
axes[1].set_ylabel("image count")
for i, v in enumerate(lc.values): axes[1].text(i, v+1, str(v), ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG/"fig1_dataset_summary.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# FIG 2 — segmentation statistics
# ---------------------------------------------------------------------------
n_rev = len(rank); n_fb = int((rank["ok"] == 0).sum())
n_fail = 2  # quarantined (IMG 77, IMG 204)
n_border = int((rank["near_fallback"] == 1).sum())
usable = n_rev - n_fail
stats["segmentation"] = dict(reviewed=n_rev, fallbacks=n_fb, quarantined=n_fail,
                             borderline_flags=n_border, usable=usable,
                             usable_pct=round(100*usable/n_rev, 1),
                             median_area_pct=round(float(rank["area_frac"].median()*100), 2),
                             mean_area_pct=round(float(rank["area_frac"].mean()*100), 2))
rank["ratio_lbl"] = rank["group"].str.replace("ratio_", "", regex=False).str.replace("_", "/")
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0,0].hist(rank["area_frac"]*100, bins=30, color="#d98a29", edgecolor="k", alpha=.85)
ax[0,0].set_title("ROI area as % of image (238 images)"); ax[0,0].set_xlabel("ROI area (%)"); ax[0,0].set_ylabel("count")
ax[0,0].axvline(rank["area_frac"].median()*100, color="navy", ls="--", label=f"median {rank['area_frac'].median()*100:.1f}%")
ax[0,0].legend()
# area% by lighting
order_l = ["ambient","red","green","blue","yellow","cycling"]
data = [rank[rank["lighting"]==l]["area_frac"].values*100 for l in order_l if (rank["lighting"]==l).any()]
labs = [l for l in order_l if (rank["lighting"]==l).any()]
ax[0,1].boxplot(data, tick_labels=labs); ax[0,1].set_title("ROI area% by lighting"); ax[0,1].set_ylabel("ROI area (%)")
# quality counts
q = {"ok\n(reviewed)": usable-n_border, "borderline": n_border, "quarantined\n(failure)": n_fail, "fallback": n_fb}
ax[1,0].bar(q.keys(), q.values(), color=["#4a9","#e2a", "#c33","#777"])
ax[1,0].set_title("Segmentation outcome (all reviewed images)"); ax[1,0].set_ylabel("count")
for i,v in enumerate(q.values()): ax[1,0].text(i, v+1, str(v), ha="center")
# confidence distribution
ax[1,1].hist(rank["confidence"], bins=25, color="#6b4f2a", edgecolor="k", alpha=.85)
ax[1,1].set_title("Segmentation confidence score"); ax[1,1].set_xlabel("confidence (100=certain)"); ax[1,1].set_ylabel("count")
plt.tight_layout(); plt.savefig(FIG/"fig2_segmentation.png", dpi=150); plt.close()

# ---------------------------------------------------------------------------
# FIG 3 — color vs ratio (AMBIENT ONLY, where color = powder not screen)
# ---------------------------------------------------------------------------
amb = feat[feat["illum"] == "ambient"].copy()
amb_ratios = [r for r in RATIOS if (amb["ratio"] == r).any()]
def series(col):
    m = [amb[amb["ratio"]==r][col].mean() for r in amb_ratios]
    s = [amb[amb["ratio"]==r][col].std(ddof=0) for r in amb_ratios]
    return np.array(m), np.array(s)
xr = np.arange(len(amb_ratios)); xl = [rlabel(r) for r in amb_ratios]
ncounts = [int((amb["ratio"]==r).sum()) for r in amb_ratios]
stats["ambient_n_per_ratio"] = dict(zip(xl, ncounts))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for col, c in [("mean_R","#c0392b"),("mean_G","#27ae60"),("mean_B","#2980b9")]:
    m, s = series(col); ax[0].errorbar(xr, m, yerr=s, marker="o", capsize=3, label=col, color=c)
ax[0].set_xticks(xr); ax[0].set_xticklabels(xl, rotation=45, ha="right")
ax[0].set_title("Mean RGB of sample vs adulteration (ambient only)"); ax[0].set_ylabel("channel value (0-255)"); ax[0].legend()
for col, c in [("mean_H","#8e44ad"),("mean_S","#e67e22"),("mean_V","#16a085")]:
    m, s = series(col); ax[1].errorbar(xr, m, yerr=s, marker="s", capsize=3, label=col, color=c)
ax[1].set_xticks(xr); ax[1].set_xticklabels(xl, rotation=45, ha="right")
ax[1].set_title("Mean HSV of sample vs adulteration (ambient only)"); ax[1].set_ylabel("HSV value"); ax[1].legend()
plt.figtext(0.5, -0.03, f"Ambient n per ratio: {dict(zip(xl,ncounts))}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG/"fig3_color_by_ratio.png", dpi=150, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------------------
# FIG 4 — PCA
# ---------------------------------------------------------------------------
X = feat[FEATURES].values
Xs = StandardScaler().fit_transform(X)
p_all = PCA(n_components=2).fit(Xs); Z = p_all.transform(Xs)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# A: all train+val colored by lighting -> shows lighting dominates
cmap = {"ambient":"#555","red":"#c0392b","green":"#27ae60","blue":"#2980b9","yellow":"#d4ac0d","cycling":"#8e44ad"}
for l in cmap:
    m = feat["illum"]==l
    if m.any(): ax[0].scatter(Z[m.values,0], Z[m.values,1], s=28, c=cmap[l], label=l, alpha=.75, edgecolors="none")
ax[0].set_title(f"PCA of all train+val (colored by LIGHTING)\nPC1 {p_all.explained_variance_ratio_[0]*100:.0f}% / PC2 {p_all.explained_variance_ratio_[1]*100:.0f}%")
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(title="illumination", fontsize=8)
# B: ambient only colored by class -> the real signal gate
Xa = amb[FEATURES].values; Xas = StandardScaler().fit_transform(Xa)
pa = PCA(n_components=2).fit(Xas); Za = pa.transform(Xas)
for cl, col, lab in [(0,"#e67e22","pure"),(1,"#6b4f2a","adulterated")]:
    m = amb["cls"].values==cl
    ax[1].scatter(Za[m,0], Za[m,1], s=45, c=col, label=lab, alpha=.8, edgecolors="k", linewidths=.4)
ax[1].set_title(f"PCA of AMBIENT only (colored by CLASS) — signal gate\nPC1 {pa.explained_variance_ratio_[0]*100:.0f}% / PC2 {pa.explained_variance_ratio_[1]*100:.0f}%")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG/"fig4_pca.png", dpi=150); plt.close()
stats["pca"] = dict(all_pc1=round(float(p_all.explained_variance_ratio_[0]),3),
                    all_pc2=round(float(p_all.explained_variance_ratio_[1]),3),
                    ambient_pc1=round(float(pa.explained_variance_ratio_[0]),3),
                    ambient_pc2=round(float(pa.explained_variance_ratio_[1]),3),
                    ambient_n=int(len(amb)))

# ---------------------------------------------------------------------------
# FIG 5 — lighting diversity: did colored light expand the feature space?
# ---------------------------------------------------------------------------
feat["light_group"] = feat["illum"].map(lambda x: "ambient" if x=="ambient" else "colored (SVI)")
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
# hue distribution
for g,c in [("ambient","#555"),("colored (SVI)","#c07b3a")]:
    d = feat[feat["light_group"]==g]["mean_H"]
    ax[0].hist(d, bins=20, alpha=.6, label=f"{g} (n={len(d)})", color=c, density=True)
ax[0].set_title("Sample mean HUE distribution"); ax[0].set_xlabel("mean hue"); ax[0].legend()
# feature-space spread: std of standardized features per group
sp = {}
for g in ["ambient","colored (SVI)"]:
    sub = feat[feat["light_group"]==g][FEATURES]
    sp[g] = float(np.mean(np.std(StandardScaler().fit_transform(feat[FEATURES].values), axis=0)))  # ref
# better: total variance in original units per group (mean of per-feature std)
amb_std = feat[feat["light_group"]=="ambient"][FEATURES].std().mean()
col_std = feat[feat["light_group"]=="colored (SVI)"][FEATURES].std().mean()
ax[1].bar(["ambient","colored (SVI)"], [amb_std, col_std], color=["#555","#c07b3a"])
ax[1].set_title("Mean within-group feature spread\n(avg per-feature std, raw units)"); ax[1].set_ylabel("avg std")
for i,v in enumerate([amb_std,col_std]): ax[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")
# PCA scatter ambient vs colored on shared axes
for g,c in [("ambient","#555"),("colored (SVI)","#c07b3a")]:
    m = (feat["light_group"]==g).values
    ax[2].scatter(Z[m,0], Z[m,1], s=25, c=c, label=g, alpha=.6, edgecolors="none")
ax[2].set_title("Feature-space coverage (PCA)\nambient vs colored"); ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2"); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG/"fig5_lighting.png", dpi=150); plt.close()
stats["lighting"] = dict(ambient_spread=round(float(amb_std),2), colored_spread=round(float(col_std),2),
                         spread_ratio=round(float(col_std/amb_std),2))

# ---------------------------------------------------------------------------
# FIG 6 — class balance evolution
# ---------------------------------------------------------------------------
def cls_counts(df, is_img_col=None):
    return df

raw_img = meta[meta["is_img"] & (meta["type"]!="starch")]
raw_pure = int((raw_img["type"]=="pure").sum()); raw_adul = int((raw_img["type"]=="yellow_dye").sum())
# after quarantine (2 adulterated removed)
qp, qa = raw_pure, raw_adul - 2
# split (images) by class
man["is_img"] = man["rel_path"].map(is_img)
mi = man[man["is_img"]]
def sc(split, cl): return int(((mi["split"]==split) & (mi["cls"]==str(cl))).sum())
tr_p, tr_a = sc("train",0), sc("train",1)
va_p, va_a = sc("val",0), sc("val",1)
te_p, te_a = sc("test",0), sc("test",1)
# FINAL counts INCLUDING frames, read straight from the assembled feature tables (real,
# not projected — reflects the frozen, balanced dataset).
def n(df, cl): return int((df["cls"] == cl).sum())
tr_pf, tr_af = n(feat[feat.split=="train"], 0), n(feat[feat.split=="train"], 1)
va_pf, va_af = n(feat[feat.split=="val"], 0),   n(feat[feat.split=="val"], 1)
te_pf, te_af = n(te_feat, 0), n(te_feat, 1)
stats["class_balance"] = dict(raw_images=dict(pure=raw_pure,adul=raw_adul),
    train_images=dict(pure=tr_p,adul=tr_a),
    train_final=dict(pure=tr_pf,adul=tr_af), val_final=dict(pure=va_pf,adul=va_af),
    test_final=dict(pure=te_pf,adul=te_af))

stages = ["raw\nimages","train\nimages","train\n+frames","val\n+frames","test\n+frames"]
pure_v = [raw_pure, tr_p, tr_pf, va_pf, te_pf]
adul_v = [raw_adul, tr_a, tr_af, va_af, te_af]
fig, ax = plt.subplots(figsize=(13,6))
x = np.arange(len(stages)); w=.42
ax.bar(x-w/2, pure_v, w, label="pure", color="#e67e22")
ax.bar(x+w/2, adul_v, w, label="adulterated", color="#6b4f2a")
for i,(p,a) in enumerate(zip(pure_v,adul_v)):
    ax.text(i-w/2, p, str(p), ha="center", va="bottom", fontsize=8)
    ax.text(i+w/2, a, str(a), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(stages); ax.set_ylabel("sample count")
ax.set_title("Class balance through the pipeline (test frames not augmented; frame counts projected)")
ax.legend(); plt.tight_layout(); plt.savefig(FIG/"fig6_class_balance.png", dpi=150); plt.close()

json.dump(stats, open(ROOT/"eda/stats.json","w"), indent=2)
print("figures written to eda/figures/:")
for p in sorted(FIG.glob("*.png")): print("  ", p.name)
print("\nkey stats:")
print("  seg usable %:", stats["segmentation"]["usable_pct"])
print("  PCA ambient PC1/PC2:", stats["pca"]["ambient_pc1"], stats["pca"]["ambient_pc2"], "n=", stats["pca"]["ambient_n"])
print("  lighting spread ratio (colored/ambient):", stats["lighting"]["spread_ratio"])
print("  train final (incl frames):", stats["class_balance"]["train_final"])
