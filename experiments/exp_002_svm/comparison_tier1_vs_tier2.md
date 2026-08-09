# Tier-1 (LogReg, 11 colour stats) vs Tier-2 (SVM, 96-d RGB histograms)

Image-only, sealed test (12 pure : 103 adulterated). Positive = adulterated. PROVISIONAL (pre-frames).

| model | variant | threshold | bal_acc | PR-AUC | pure_recall | adul_recall | accuracy |
|---|---|---|---|---|---|---|---|
| Tier-1 LogReg | unweighted | 0.5 | 0.673 | 0.679 | 0.81 | 0.54 | 0.672 |
| Tier-1 LogReg | unweighted | val-thr | 0.637 | 0.679 | 0.54 | 0.73 | 0.638 |
| Tier-1 LogReg | balanced | 0.5 | 0.673 | 0.679 | 0.81 | 0.54 | 0.672 |
| Tier-1 LogReg | balanced | val-thr | 0.637 | 0.679 | 0.54 | 0.73 | 0.638 |
| Tier-2 SVM | unweighted | 0.5 | 0.766 | 0.874 | 0.85 | 0.69 | 0.766 |
| Tier-2 SVM | unweighted | val-thr | 0.765 | 0.874 | 0.70 | 0.83 | 0.765 |
| Tier-2 SVM | balanced | 0.5 | 0.767 | 0.874 | 0.85 | 0.69 | 0.767 |
| Tier-2 SVM | balanced | val-thr | 0.763 | 0.874 | 0.70 | 0.83 | 0.763 |
