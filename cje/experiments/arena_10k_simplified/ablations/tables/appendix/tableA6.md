## Table A6: Runtime & Complexity

| Estimator                                             |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:------------------------------------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips (iic=False)                                   |         0.263229 |      0.37373  | O(n)         |         0 |         0.11699  |            nan |
| calibrated-ips (iic=False)                            |         0.341278 |      0.551725 | O(n)         |         0 |         0.151679 |            nan |
| calibrated-ips (iic=True)                             |         0.365823 |      0.555945 | O(n)         |         0 |         0.162588 |            nan |
| orthogonalized-ips (iic=False)                        |         0.412599 |      0.624968 | O(n)         |         0 |         0.183377 |            nan |
| raw-ips (iic=True)                                    |         1.10376  |      2.46876  | O(n)         |         0 |         0.490561 |            nan |
| orthogonalized-ips (iic=True)                         |         1.26222  |      2.70959  | O(n)         |         0 |         0.560988 |            nan |
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
| dr-cpo (calib=False, iic=False)                       |         8.52569  |     42.337    | O(K*n)       |        20 |         3.7892   |            nan |
| dr-cpo (calib=True, iic=False)                        |         8.58217  |     42.5915   | O(K*n)       |        20 |         3.8143   |            nan |
| dr-cpo (calib=False, iic=True)                        |         9.63394  |     44.5232   | O(K*n)       |        20 |         4.28175  |            nan |
| dr-cpo (calib=True, iic=True)                         |         9.65517  |     44.6284   | O(K*n)       |        20 |         4.29119  |            nan |
| stacked-dr (iic=False)                                |        16.686    |    222.237    | O(M*K*n)     |        20 |         7.92587  |              5 |
| stacked-dr (iic=True)                                 |        16.8327   |    227.612    | O(M*K*n)     |        20 |         7.99555  |              5 |
| stacked-dr-core (calib=True, iic=False)               |        39.0154   |    104.958    | O(M*K*n)     |        20 |        17.3402   |              4 |
| stacked-dr-core (calib=True, iic=True)                |        39.6962   |    106.871    | O(M*K*n)     |        20 |        17.6428   |              4 |