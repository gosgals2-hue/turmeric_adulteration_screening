# exp_001 — Tier-1 Logistic Regression (colour features)

Baseline classical model for pure-vs-adulterated (yellow dye) turmeric screening,
on masked colour/texture features. **Image-only** (video frames not yet added).
Trained on TRAIN, threshold chosen on VAL, reported on the **sealed TEST** set.
Positive class = adulterated. See `config.json` for the full protocol.

## Test results

Header: accuracy · balanced accuracy · PR-AUC(adulterated) · pure recall · adulterated recall · macro-F1

**@ default threshold 0.5**

| model | acc | bal_acc | PR-AUC | pure_recall | adul_recall | macroF1 |
|---|---|---|---|---|---|---|
| unweighted | 0.844 | 0.508 | 0.968 | 0.08 | 0.93 | 0.51 |
| balanced | 0.565 | 0.647 | 0.968 | 0.75 | 0.54 | 0.48 |

**@ val-selected threshold (maximises balanced accuracy on val)**

| model | acc | bal_acc | PR-AUC | pure_recall | adul_recall | macroF1 |
|---|---|---|---|---|---|---|
| unweighted | 0.461 | 0.699 | 0.968 | 1.00 | 0.40 | 0.42 |
| balanced | 0.487 | 0.714 | 0.968 | 1.00 | 0.43 | 0.44 |

## Read

- **Ranking signal is real:** PR-AUC ≈ 0.968 for adulterated, though the no-skill
  baseline is 0.90 (prevalence), so the honest lift is modest.
- **Unweighted @0.5 is the imbalance trap:** 84% accuracy but 51% balanced accuracy and
  8% pure recall — it mostly predicts "adulterated". This is exactly why accuracy alone
  is misleading here.
- **`class_weight='balanced'` helps the minority:** pure recall 0.08 → 0.75 at 0.5,
  balanced accuracy 0.51 → 0.65.
- **Threshold tuning trades off hard:** pushing pure recall to 1.0 (val threshold) collapses
  adulterated recall to ~0.4 and accuracy below 0.5. With only 12 val samples the chosen
  threshold is unreliable — treat these operating points as indicative, not final.

## Next levers
- Add dense **pure video frames** to enrich the minority class (currently 5 pure piles only).
- Tier-2 (SVM on colour histograms), Tier-3 (MobileNetV2) for comparison.
- Report per-ratio sensitivity (esp. low adulteration) once frames are in.
