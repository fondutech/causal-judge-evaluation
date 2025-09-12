## Table A6: Runtime & Complexity

| Estimator                       |   Runtime_Median |   Runtime_P90 | Complexity   |   N_Folds |   Runtime_per_1k |   M_Components |
|:--------------------------------|-----------------:|--------------:|:-------------|----------:|-----------------:|---------------:|
| raw-ips (iic=False)             |         0.244369 |      0.366926 | O(n)         |         0 |         0.108608 |            nan |
| calibrated-ips (iic=False)      |         0.314293 |      0.517315 | O(n)         |         0 |         0.139686 |            nan |
| calibrated-ips (iic=True)       |         0.33112  |      0.518463 | O(n)         |         0 |         0.147164 |            nan |
| raw-ips (iic=True)              |         0.345294 |      0.526655 | O(n)         |         0 |         0.153464 |            nan |
| orthogonalized-ips (iic=False)  |         0.358995 |      0.605536 | O(n)         |         0 |         0.159553 |            nan |
| orthogonalized-ips (iic=True)   |         0.45514  |      0.785457 | O(n)         |         0 |         0.202284 |            nan |
| tr-cpo-e (iic=False)            |        19.9752   |    104.933    | O(K*n)       |        20 |         8.87789  |            nan |
| tr-cpo (iic=False)              |        20.0031   |    103.634    | O(K*n)       |        20 |         8.89026  |            nan |
| tr-cpo (iic=True)               |        20.0873   |    102.641    | O(K*n)       |        20 |         8.92767  |            nan |
| tr-cpo-e (iic=True)             |        20.1011   |    104.602    | O(K*n)       |        20 |         8.93384  |            nan |
| oc-dr-cpo (iic=False)           |        20.8141   |    103.834    | O(K*n)       |        20 |         9.25072  |            nan |
| oc-dr-cpo (iic=True)            |        20.9901   |    102.776    | O(K*n)       |        20 |         9.32893  |            nan |
| dr-cpo (calib=False, iic=False) |        21.9362   |    108.218    | O(K*n)       |        20 |         9.74943  |            nan |
| dr-cpo (calib=False, iic=True)  |        22.0675   |    107.995    | O(K*n)       |        20 |         9.80779  |            nan |
| dr-cpo (calib=True, iic=False)  |        22.0941   |    108.484    | O(K*n)       |        20 |         9.81961  |            nan |
| dr-cpo (calib=True, iic=True)   |        22.3508   |    107.685    | O(K*n)       |        20 |         9.93371  |            nan |
| stacked-dr (iic=True)           |        62.7667   |    309.781    | O(M*K*n)     |        20 |        27.8963   |              5 |
| stacked-dr (iic=False)          |        62.8024   |    309.736    | O(M*K*n)     |        20 |        27.9122   |              5 |