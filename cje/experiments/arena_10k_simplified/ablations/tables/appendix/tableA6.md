## Table A6: Runtime & Complexity

| Estimator                                             |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:------------------------------------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips (iic=False)                                   |         0.312087 |      0.551336 | O(n)         |         0 |         0.138705 |            nan |
| calibrated-ips (iic=False)                            |         0.352787 |      0.71971  | O(n)         |         0 |         0.156794 |            nan |
| orthogonalized-ips (iic=False)                        |         0.417406 |      0.805085 | O(n)         |         0 |         0.185514 |            nan |
| calibrated-ips (iic=True)                             |         0.467758 |      0.693113 | O(n)         |         0 |         0.207893 |            nan |
| raw-ips (iic=True)                                    |         1.22875  |      3.37969  | O(n)         |         0 |         0.54611  |            nan |
| orthogonalized-ips (iic=True)                         |         1.34017  |      3.63185  | O(n)         |         0 |         0.595633 |            nan |
| tr-cpo-e (iic=False)                                  |         6.93965  |     31.6988   | O(K*n)       |        20 |         3.08429  |            nan |
| tr-cpo-e-anchored-orthogonal (calib=False, iic=False) |         7.0378   |     31.8777   | O(K*n)       |        20 |         3.12791  |            nan |
| tr-cpo-e-anchored (calib=False, iic=False)            |         7.0501   |     31.736    | O(K*n)       |        20 |         3.13338  |            nan |
| tr-cpo (iic=False)                                    |         7.12576  |     31.6757   | O(K*n)       |        20 |         3.16701  |            nan |
| oc-dr-cpo (iic=False)                                 |         7.74426  |     35.4318   | O(K*n)       |        20 |         3.44189  |            nan |
| tr-cpo-e (iic=True)                                   |         8.01793  |     33.8954   | O(K*n)       |        20 |         3.56352  |            nan |
| tr-cpo-e-anchored (calib=False, iic=True)             |         8.03079  |     33.8514   | O(K*n)       |        20 |         3.56924  |            nan |
| tr-cpo (iic=True)                                     |         8.03574  |     33.828    | O(K*n)       |        20 |         3.57144  |            nan |
| tr-cpo-e-anchored-orthogonal (calib=False, iic=True)  |         8.18644  |     34.1322   | O(K*n)       |        20 |         3.63842  |            nan |
| oc-dr-cpo (iic=True)                                  |         8.69561  |     37.6068   | O(K*n)       |        20 |         3.86472  |            nan |
| dr-cpo (calib=False, iic=False)                       |         8.75173  |     42.6194   | O(K*n)       |        20 |         3.88966  |            nan |
| dr-cpo (calib=True, iic=False)                        |         8.82556  |     42.8177   | O(K*n)       |        20 |         3.92247  |            nan |
| dr-cpo (calib=False, iic=True)                        |         9.79373  |     44.7246   | O(K*n)       |        20 |         4.35277  |            nan |
| dr-cpo (calib=True, iic=True)                         |         9.91366  |     44.9294   | O(K*n)       |        20 |         4.40607  |            nan |
| stacked-dr-core (calib=True, iic=False)               |        39.8624   |    106.065    | O(M*K*n)     |        20 |        17.7166   |              4 |
| stacked-dr-core (calib=True, iic=True)                |        40.5059   |    108.007    | O(M*K*n)     |        20 |        18.0026   |              4 |
| stacked-dr (iic=False)                                |        46.4277   |    225.45     | O(M*K*n)     |        20 |        20.6346   |              5 |
| stacked-dr (iic=True)                                 |        49.1612   |    231.776    | O(M*K*n)     |        20 |        21.8494   |              5 |