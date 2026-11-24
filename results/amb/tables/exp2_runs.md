Per-run individual accuracy (%) (exploratory, post hoc). take-off = definition individual accuracy > 70%. k=1 / (def/2): ratio of k=1 accuracy to half the definition accuracy (choosing one of the two consistent operations at random).

| arch | seed | definition | k=1 | identifiable | not identifiable | k=1 / (def/2) | take-off |
|---|---|---|---|---|---|---|---|
| hybrid_early | 0 | 46.8 | 34.5 | 43.1 | 35.2 | 1.48 | no |
| hybrid_early | 1 | 46.8 | 35.6 | 44.7 | 36.9 | 1.52 | no |
| hybrid_early | 2 | 43.7 | 33.4 | 40.9 | 34.9 | 1.53 | no |
| hybrid_early | 3 | 49.1 | 35.9 | 44.4 | 37.3 | 1.46 | no |
| hybrid_early | 4 | 50.6 | 35.4 | 46.4 | 36.9 | 1.40 | no |
| hybrid_late | 0 | 44.6 | 34.2 | 41.4 | 35.2 | 1.53 | no |
| hybrid_late | 1 | 85.5 | 46.3 | 84.2 | 46.4 | 1.08 | yes |
| hybrid_late | 2 | 46.5 | 35.1 | 43.2 | 37.0 | 1.51 | no |
| hybrid_late | 3 | 45.2 | 33.6 | 43.0 | 35.1 | 1.49 | no |
| hybrid_late | 4 | 92.5 | 47.6 | 92.4 | 48.3 | 1.03 | yes |
| hybrid_middle | 0 | 73.9 | 43.9 | 73.2 | 44.0 | 1.19 | yes |
| hybrid_middle | 1 | 92.2 | 47.2 | 92.2 | 46.8 | 1.02 | yes |
| hybrid_middle | 2 | 48.5 | 36.1 | 45.6 | 37.5 | 1.49 | no |
| hybrid_middle | 3 | 50.5 | 37.3 | 46.6 | 38.8 | 1.48 | no |
| hybrid_middle | 4 | 46.1 | 35.9 | 43.4 | 38.0 | 1.56 | no |
| pure | 0 | 44.3 | 35.6 | 42.0 | 36.8 | 1.61 | no |
| pure | 1 | 50.2 | 36.6 | 46.1 | 37.4 | 1.46 | no |
| pure | 2 | 47.8 | 36.7 | 44.8 | 38.4 | 1.54 | no |
| pure | 3 | 47.9 | 35.4 | 43.8 | 36.8 | 1.48 | no |
| pure | 4 | 89.9 | 45.4 | 88.4 | 45.4 | 1.01 | yes |

Take-off rate 5/20. Seeds in which all 4 architectures agree: 2/5 (expected under independence at this rate: 1.6; per-seed probability 0.32).
