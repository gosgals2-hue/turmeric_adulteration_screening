# Deployment metrics — prevalence sweep (Tier-2 SVM (96-d histograms))

class_weight=balanced. Threshold chosen on VAL for target specificity 0.95 -> test Sp=0.81, overall test Se=0.73 (pile-bootstrap 95% CI 0.542–0.932). ROC-AUC=0.855, PR-AUC=0.874. Specificity rests on 5 pure pile(s).

| scenario (pure:adul) | PPV | NPV | alert rate | weighted acc | per-1000: detected / missed / false-alarms |
|---|---|---|---|---|---|
| 99:1 | 0.04 | 0.997 | 0.195 | 0.809 | 7 / 3 / 188 |
| 95:5 | 0.17 | 0.983 | 0.217 | 0.806 | 37 / 13 / 180 |
| 90:9 | 0.30 | 0.964 | 0.244 | 0.802 | 73 / 27 / 171 |

## Se@Sp operating points (threshold from val)
| target Sp | threshold | test Sp | overall test Se |
|---|---|---|---|
| 0.9 | 0.184 | 0.686 | 0.84 |
| 0.95 | 0.394 | 0.81 | 0.731 |
| 0.98 | 0.747 | 0.949 | 0.566 |
