## Table A6: Runtime & Complexity

| Estimator                                             |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:------------------------------------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips (iic=False)                                   |         0.276884 |      0.403668 | O(n)         |         0 |         0.12306  |            nan |
| calibrated-ips (iic=False)                            |         0.344015 |      0.595861 | O(n)         |         0 |         0.152896 |            nan |
| calibrated-ips (iic=True)                             |         0.373352 |      0.591655 | O(n)         |         0 |         0.165934 |            nan |
| orthogonalized-ips (iic=False)                        |         0.412599 |      0.705697 | O(n)         |         0 |         0.183377 |            nan |
| raw-ips (iic=True)                                    |         1.11038  |      2.39106  | O(n)         |         0 |         0.493503 |            nan |
| orthogonalized-ips (iic=True)                         |         1.21312  |      2.67031  | O(n)         |         0 |         0.539166 |            nan |
| dr-cpo (calib=False, iic=False)                       |         3.35054  |     42.2626   | O(K*n)       |        20 |         1.67527  |            nan |
| dr-cpo (calib=False, iic=True)                        |         3.39161  |     44.5135   | O(K*n)       |        20 |         1.65157  |            nan |
| dr-cpo (calib=True, iic=False)                        |         3.39754  |     42.5411   | O(K*n)       |        20 |         1.65446  |            nan |
| dr-cpo (calib=True, iic=True)                         |         3.45677  |     44.5955   | O(K*n)       |        20 |         1.6833   |            nan |
| tr-cpo (iic=False)                                    |         6.73708  |     31.3472   | O(K*n)       |        20 |         2.99426  |            nan |
| tr-cpo-e (iic=False)                                  |         6.77763  |     31.5214   | O(K*n)       |        20 |         3.01228  |            nan |
| tr-cpo-e-anchored (calib=False, iic=False)            |         6.84211  |     31.612    | O(K*n)       |        20 |         3.04094  |            nan |
| tr-cpo-e-anchored-orthogonal (calib=False, iic=False) |         6.85784  |     31.9871   | O(K*n)       |        20 |         3.04793  |            nan |
| oc-dr-cpo (iic=False)                                 |         7.4377   |     35.222    | O(K*n)       |        20 |         3.30565  |            nan |
| tr-cpo (iic=True)                                     |         7.79095  |     33.531    | O(K*n)       |        20 |         3.46264  |            nan |
| tr-cpo-e (iic=True)                                   |         7.81532  |     33.9442   | O(K*n)       |        20 |         3.47348  |            nan |
| tr-cpo-e-anchored (calib=False, iic=True)             |         7.87827  |     33.9583   | O(K*n)       |        20 |         3.50146  |            nan |
| tr-cpo-e-anchored-orthogonal (calib=False, iic=True)  |         7.91381  |     34.2202   | O(K*n)       |        20 |         3.51725  |            nan |
| oc-dr-cpo (iic=True)                                  |         8.37439  |     37.3413   | O(K*n)       |        20 |         3.72195  |            nan |
| stacked-dr-core (calib=True, iic=False)               |        39.3776   |    105.421    | O(M*K*n)     |        20 |        17.5011   |              4 |
| stacked-dr-core (calib=True, iic=True)                |        40.0159   |    107.974    | O(M*K*n)     |        20 |        17.7848   |              4 |
| stacked-dr (iic=False)                                |        46.4621   |    226.176    | O(M*K*n)     |        20 |        20.6498   |              5 |
| stacked-dr (iic=True)                                 |        49.1045   |    232.725    | O(M*K*n)     |        20 |        21.8242   |              5 |