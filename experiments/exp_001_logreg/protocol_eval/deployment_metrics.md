# Deployment metrics — prevalence sweep (Tier-1 LogReg (11 colour stats))

class_weight=balanced. Threshold chosen on VAL for target specificity 0.95 -> test Sp=0.82, overall test Se=0.49 (pile-bootstrap 95% CI 0.345–0.651). ROC-AUC=0.698, PR-AUC=0.679. Specificity rests on 5 pure pile(s).

| scenario (pure:adul) | PPV | NPV | alert rate | weighted acc | per-1000: detected / missed / false-alarms |
|---|---|---|---|---|---|
| 99:1 | 0.03 | 0.994 | 0.178 | 0.822 | 5 / 5 / 173 |
| 95:5 | 0.13 | 0.969 | 0.191 | 0.808 | 25 / 25 / 166 |
| 90:9 | 0.24 | 0.936 | 0.207 | 0.792 | 49 / 51 / 158 |

## Se@Sp operating points (threshold from val)
| target Sp | threshold | test Sp | overall test Se |
|---|---|---|---|
| 0.9 | 0.403 | 0.651 | 0.689 |
| 0.95 | 0.525 | 0.825 | 0.491 |
| 0.98 | 0.652 | 0.897 | 0.277 |
