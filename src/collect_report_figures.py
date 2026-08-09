"""
collect_report_figures.py — Copy result figures into report/figures/ with descriptive,
numbered names matching report/FIGURE_MANIFEST.md.

Copies (never moves), so the scripts that generate the originals keep working and can be
re-run freely. Re-run this after eda_figures.py / train_tier3_cnn.py / evaluate_protocol.py
to refresh the renamed copies. Missing sources are skipped with a note.

Run from project root:  python src/collect_report_figures.py
"""
import shutil
from pathlib import Path

ROOT = Path.cwd()
for _ in range(4):
    if (ROOT / "data" / "metadata.csv").exists():
        break
    ROOT = ROOT.parent
OUT = ROOT / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

# source (relative to ROOT) -> descriptive report filename
MAP = {
    # Signal check / EDA (run eda_figures.py first for current versions)
    "eda/figures/fig1_dataset_summary.png": "fig01_dataset_summary.png",
    "eda/figures/fig3_color_by_ratio.png": "fig02_color_signal_by_ratio.png",
    "eda/figures/fig4_pca.png": "fig03_pca_lighting_vs_class.png",
    "eda/figures/fig5_lighting.png": "fig04_ambient_vs_svi_spread.png",
    "eda/figures/fig2_segmentation.png": "fig05_segmentation_qa.png",
    # Model comparison
    "experiments/exp_001_logreg/pr_and_metrics.png": "fig06_tier1_logreg_pr.png",
    "experiments/exp_002_svm/pr_and_metrics.png": "fig07_tier2_svm_pr.png",
    "experiments/exp_003_cnn/pr_and_metrics.png": "fig08_tier3_cnn_pr.png",
    # Limit of detection (per-ratio sensitivity + ROC)
    "experiments/exp_002_svm/protocol_eval/sensitivity_and_roc.png": "fig09_tier2_limit_of_detection.png",
    "experiments/exp_003_cnn/protocol_eval/sensitivity_and_roc.png": "fig10_tier3_limit_of_detection.png",
    # Degradation robustness
    "experiments/exp_degradation/heatmap_cnn.png": "fig11_tier3_degradation_heatmap.png",
    "experiments/exp_degradation/curves_cnn_brightness.png": "fig12_tier3_brightness_curve.png",
    "experiments/exp_degradation/curves_cnn_blur.png": "fig13_tier3_blur_curve.png",
    # Cost-benefit
    "report/figures/cba_triage_flow.png": "fig14_cba_triage_flow.png",
    # Supplementary
    "experiments/exp_001_logreg/confusion.png": "figS1_tier1_confusion.png",
    "experiments/exp_001_logreg/protocol_eval/sensitivity_and_roc.png": "figS2_tier1_limit_of_detection.png",
    "experiments/exp_degradation/heatmap_tier1.png": "figS3_tier1_degradation_heatmap.png",
    "experiments/exp_degradation/heatmap_tier2.png": "figS4_tier2_degradation_heatmap.png",
    "experiments/exp_degradation/curves_cnn_contrast.png": "figS5_tier3_contrast_curve.png",
    "experiments/exp_degradation/curves_cnn_noise.png": "figS6_tier3_noise_curve.png",
    "experiments/exp_degradation/curves_cnn_downscale.png": "figS7_tier3_downscale_curve.png",
    "experiments/exp_degradation/curves_cnn_jpeg.png": "figS8_tier3_jpeg_curve.png",
}


def main():
    copied = skipped = 0
    for src_rel, dst_name in MAP.items():
        src = ROOT / src_rel
        if src_rel.startswith("report/figures/") and (ROOT / "report/figures" / dst_name) == src:
            continue
        if not src.exists():
            print(f"  skip (missing): {src_rel}"); skipped += 1; continue
        shutil.copy2(src, OUT / dst_name)
        print(f"  {dst_name:38} <- {src_rel}"); copied += 1
    print(f"\nCopied {copied} figures to {OUT} ({skipped} missing/skipped).")


if __name__ == "__main__":
    main()
