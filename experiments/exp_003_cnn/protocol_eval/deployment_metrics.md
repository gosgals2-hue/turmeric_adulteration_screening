# Deployment metrics — prevalence sweep (Tier-3 MobileNetV2 (CNN))

class_weight=balanced. Threshold chosen on VAL for target specificity 0.95 -> test Sp=0.89, overall test Se=0.89 (pile-bootstrap 95% CI 0.763–0.986). ROC-AUC=0.952, PR-AUC=0.944. Specificity rests on 5 pure pile(s).

| scenario (pure:adul) | PPV | NPV | alert rate | weighted acc | per-1000: detected / missed / false-alarms |
|---|---|---|---|---|---|
| 99:1 | 0.07 | 0.999 | 0.119 | 0.889 | 9 / 1 / 110 |
| 95:5 | 0.30 | 0.993 | 0.150 | 0.889 | 44 / 6 / 105 |
| 90:9 | 0.47 | 0.986 | 0.189 | 0.889 | 89 / 11 / 100 |

## Se@Sp operating points (threshold from val)
| target Sp | threshold | test Sp | overall test Se |
|---|---|---|---|
| 0.9 | 0.441 | 0.845 | 0.935 |
| 0.95 | 0.551 | 0.889 | 0.888 |
| 0.98 | 0.843 | 0.962 | 0.635 |
