# exp_002 — Tier-2 SVM (RBF) on 32-bin RGB colour histograms

Richer colour representation than Tier-1: per masked sample, 32-bin normalized R/G/B
histograms (96-dim) instead of 11 summary stats. RBF SVM (C=10, gamma=scale), same
protocol (train→val threshold→sealed test), unweighted + balanced. Positive = adulterated.
Image-only, **provisional** (pre-frames, single pure test pile).

## Headline
Tier-2 substantially beats Tier-1 on the minority class. Best config (balanced, val
threshold): **balanced accuracy 0.84, pure recall 0.83, adulterated recall 0.85** — vs
Tier-1's best balanced accuracy 0.71 that only reached pure recall 1.0 by collapsing
adulterated recall to 0.43. Tier-2 also lifts PR-AUC 0.968→0.99 and gives a usable ROC-AUC 0.93.

See `comparison_tier1_vs_tier2.md` for the full table.

## Read
- The colour *distribution* (histogram) carries much more separable signal than colour
  *means* — expected, since yellow dye reshapes the histogram rather than just shifting a mean.
- The SVM no longer needs an extreme threshold to detect pure: even at 0.5 it reaches
  pure recall 0.67 / bal-acc 0.82, and balanced+val-threshold trades to a genuine 0.83/0.85.
- Same caveats hold: 12-sample val makes the threshold noisy; specificity rests on one pure
  pile; numbers are pre-frame. Re-run after frames + more pure piles, then feed the best
  model to `evaluate_protocol.py` for the per-ratio + deployment analysis.

## Next
- Tier-3 (MobileNetV2 transfer learning) for the final comparison row.
- Light SVM hyper-tuning (C, gamma) on val/CV once the training set is frame-enriched.
