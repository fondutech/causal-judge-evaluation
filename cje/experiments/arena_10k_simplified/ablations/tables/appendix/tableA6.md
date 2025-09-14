## Table A6: Runtime & Complexity

| Estimator                       |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:--------------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips (iic=False)             |         0.24293  |      0.357142 | O(n)         |         0 |         0.107969 |            nan |
| raw-ips (iic=True)              |         0.277228 |      0.419791 | O(n)         |         0 |         0.123212 |            nan |
| calibrated-ips (iic=False)      |         0.327593 |      0.575381 | O(n)         |         0 |         0.145597 |            nan |
| calibrated-ips (iic=True)       |         0.337518 |      0.581537 | O(n)         |         0 |         0.150008 |            nan |
| orthogonalized-ips (iic=False)  |         0.355965 |      0.580807 | O(n)         |         0 |         0.158207 |            nan |
| orthogonalized-ips (iic=True)   |         0.391332 |      0.656374 | O(n)         |         0 |         0.173925 |            nan |
| stacked-dr (iic=False)          |         5.52341  |     14.7541   | O(M*K*n)     |        20 |         7.94596  |              5 |
| stacked-dr (iic=True)           |         5.56985  |     14.8076   | O(M*K*n)     |        20 |         7.7121   |              5 |
| tr-cpo (iic=False)              |         6.76471  |     31.3881   | O(K*n)       |        20 |         3.00654  |            nan |
| tr-cpo-e (iic=False)            |         6.79794  |     31.4214   | O(K*n)       |        20 |         3.02131  |            nan |
| tr-cpo (iic=True)               |         6.80399  |     31.5729   | O(K*n)       |        20 |         3.024    |            nan |
| tr-cpo-e (iic=True)             |         6.83183  |     31.4838   | O(K*n)       |        20 |         3.03637  |            nan |
| oc-dr-cpo (iic=False)           |         7.40275  |     35.0079   | O(K*n)       |        20 |         3.29011  |            nan |
| oc-dr-cpo (iic=True)            |         7.43733  |     35.0623   | O(K*n)       |        20 |         3.30548  |            nan |
| dr-cpo (calib=False, iic=False) |         8.45149  |     42.073    | O(K*n)       |        20 |         3.75622  |            nan |
| dr-cpo (calib=False, iic=True)  |         8.50214  |     42.1743   | O(K*n)       |        20 |         3.77873  |            nan |
| dr-cpo (calib=True, iic=False)  |         8.51603  |     42.1669   | O(K*n)       |        20 |         3.7849   |            nan |
| dr-cpo (calib=True, iic=True)   |         8.5656   |     42.236    | O(K*n)       |        20 |         3.80693  |            nan |