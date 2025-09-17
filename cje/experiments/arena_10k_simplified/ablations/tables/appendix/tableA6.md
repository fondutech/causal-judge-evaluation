## Table A6: Runtime & Complexity

| Estimator                    |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:-----------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| calibrated-ips               |          2.98976 |      10.6647  | O(n)         |         0 |          1.61608 |            nan |
| orthogonalized-ips           |          3.00613 |      10.0302  | O(n)         |         0 |          1.62494 |            nan |
| raw-ips                      |          3.01194 |       8.33586 | O(n)         |         0 |          1.62807 |            nan |
| dr-cpo                       |          3.32657 |      56.6984  | O(K*n)       |        20 |          2.38509 |            nan |
| dr-cpo (calib)               |          3.59835 |      58.464   | O(K*n)       |        20 |          2.57995 |            nan |
| oc-dr-cpo                    |         11.8219  |     104.58    | O(K*n)       |        20 |          6.3902  |            nan |
| tr-cpo-e-anchored-orthogonal |         12.5432  |      91.7561  | O(K*n)       |        20 |          6.7801  |            nan |
| tr-cpo-e                     |         12.6393  |      98.788   | O(K*n)       |        20 |          6.83207 |            nan |
| stacked-dr                   |         16.9889  |      88.8111  | O(M*K*n)     |        20 |         11.9602  |              5 |