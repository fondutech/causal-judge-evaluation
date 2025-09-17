## Table A6: Runtime & Complexity

| Estimator                    |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:-----------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips                      |          2.4134  |       3.94045 | O(n)         |         0 |          1.30454 |            nan |
| calibrated-ips               |          2.45426 |       4.58513 | O(n)         |         0 |          1.32663 |            nan |
| orthogonalized-ips           |          2.47979 |       4.31553 | O(n)         |         0 |          1.34043 |            nan |
| dr-cpo                       |          5.66234 |      51.7443  | O(K*n)       |        20 |          3.06072 |            nan |
| dr-cpo (calib)               |          5.74072 |      51.4288  | O(K*n)       |        20 |          3.10309 |            nan |
| oc-dr-cpo                    |          6.3283  |      30.3935  | O(K*n)       |        20 |          5.95605 |            nan |
| tr-cpo-e-anchored-orthogonal |          6.64072 |      59.3607  | O(K*n)       |        20 |          3.58958 |            nan |
| tr-cpo-e                     |          6.73942 |      57.5395  | O(K*n)       |        20 |          4.27605 |            nan |
| stacked-dr                   |         13.7725  |     169.313   | O(M*K*n)     |        20 |          7.44461 |              5 |