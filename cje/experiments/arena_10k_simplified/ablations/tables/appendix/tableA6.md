## Table A6: Runtime & Complexity

| Estimator                    |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:-----------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips                      |          5.81198 |       12.3987 | O(n)         |         0 |          3.14161 |            nan |
| calibrated-ips               |          6.02033 |       12.6297 | O(n)         |         0 |          3.25423 |            nan |
| orthogonalized-ips           |          6.16486 |       15.3612 | O(n)         |         0 |          3.33236 |            nan |
| tr-cpo-e                     |         10.7253  |       96.935  | O(K*n)       |        20 |          5.79747 |            nan |
| tr-cpo-e-anchored-orthogonal |         11.0391  |       93.914  | O(K*n)       |        20 |          5.96709 |            nan |
| dr-cpo                       |         13.22    |       79.2801 | O(K*n)       |        20 |          7.14593 |            nan |
| dr-cpo (calib)               |         13.9853  |       79.8652 | O(K*n)       |        20 |          7.55961 |            nan |
| oc-dr-cpo                    |         14.2508  |      104.905  | O(K*n)       |        20 |          7.70315 |            nan |
| stacked-dr                   |         17.9695  |      220.894  | O(M*K*n)     |        20 |          9.71322 |              5 |