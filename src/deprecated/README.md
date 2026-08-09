# Deprecated scripts — DO NOT USE for results

These are early exploratory scripts (June 2026), superseded by the tiered pipeline.
They are kept only for history. **None of the numbers they print are trustworthy or
reported anywhere**, because they are NOT leak-free:

- They read an old flat `features/dataset.csv` (pre-segmentation, pre-metadata).
- They use a random, file-level `train_test_split(stratify=y)` — so frames/views of the
  SAME physical pile can land in both train and test, which inflates scores.
- `train_svm.py` / `train_rf.py` also use plain `cross_val_score(cv=5)` (ordinary KFold,
  no `groups`), so their cross-validation has the same pile leak.

## Use these instead (leak-free, pile-aware)

| Deprecated            | Replacement                         |
|-----------------------|-------------------------------------|
| `train_logreg.py`     | `../train_tier1_logreg.py`          |
| `train_svm.py`        | `../train_tier2_svm.py` (GroupKFold-tuned) |
| `train_rf.py`         | (dropped — SVM/CNN cover the tiers) |
| `train_model.py`      | the tiered scripts above            |

The real pipeline splits by pile (`split_dataset.py`), extracts masked features
(`extract_features.py`, `extract_hist_features.py`), and both the final split AND the
Tier-2 hyperparameter search (`GridSearchCV` + `GroupKFold` over `pile_id`) respect pile
boundaries. Reported metrics come only from the sealed, pile-clean TEST set.
