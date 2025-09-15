|   Rank | Estimator                                             |   AggScore |   RMSE_d |   IntervalScore_OA |   CalibScore |   SE_GeoMean |   Kendall_tau |   Top1_Acc |
|-------:|:------------------------------------------------------|-----------:|---------:|-------------------:|-------------:|-------------:|--------------:|-----------:|
|      1 | dr-cpo (calib=True, iic=True)                         |       97.6 |   0.0104 |             0.1184 |          5   |       0.0213 |         0.583 |       97.5 |
|      2 | dr-cpo (calib=True, iic=False)                        |       97.5 |   0.0104 |             0.1155 |          5.6 |       0.0206 |         0.583 |       97.5 |
|      3 | stacked-dr-core (calib=True, iic=False)               |       97.4 |   0.0132 |             0.0708 |          7.8 |       0.0126 |         0.533 |      100   |
|      4 | stacked-dr-core (calib=True, iic=True)                |       97.4 |   0.0132 |             0.0711 |          7.8 |       0.0126 |         0.533 |      100   |
|      5 | stacked-dr (iic=True)                                 |       95.5 |   0.0119 |             0.0682 |          9.9 |       0.0117 |         0.505 |       94.3 |
|      6 | stacked-dr (iic=False)                                |       95.1 |   0.0122 |             0.0694 |         10   |       0.0118 |         0.49  |       94.1 |
|      7 | oc-dr-cpo (iic=True)                                  |       82.5 |   0.0373 |             0.0922 |          9   |       0.0218 |         0.183 |       65   |
|      8 | oc-dr-cpo (iic=False)                                 |       82.3 |   0.0373 |             0.0862 |         10.4 |       0.0198 |         0.183 |       65   |
|      9 | dr-cpo (calib=False, iic=False)                       |       80.5 |   0.0354 |             0.1935 |          5   |       0.0349 |         0.167 |       65   |
|     10 | dr-cpo (calib=False, iic=True)                        |       80.1 |   0.0354 |             0.2046 |          5   |       0.0369 |         0.167 |       65   |
|     11 | tr-cpo-e-anchored (calib=False, iic=False)            |       68.8 |   0.0875 |             0.1413 |         10.5 |       0.032  |        -0.083 |       35   |
|     12 | tr-cpo-e (iic=False)                                  |       68.8 |   0.0875 |             0.1413 |         10.5 |       0.032  |        -0.083 |       35   |
|     13 | tr-cpo-e (iic=True)                                   |       68.2 |   0.0875 |             0.1637 |          8.8 |       0.0386 |        -0.083 |       35   |
|     14 | tr-cpo-e-anchored (calib=False, iic=True)             |       68.2 |   0.0875 |             0.1637 |          8.8 |       0.0386 |        -0.083 |       35   |
|     15 | tr-cpo-e-anchored-orthogonal (calib=False, iic=False) |       68   |   0.1142 |             0.1841 |          9.7 |       0.0408 |        -0.05  |       40   |
|     16 | tr-cpo-e-anchored-orthogonal (calib=False, iic=True)  |       66.9 |   0.1142 |             0.2104 |          8.5 |       0.0497 |        -0.05  |       40   |
|     17 | calibrated-ips (iic=True)                             |       61.3 |   0.017  |             0.0908 |         34.2 |       0.0099 |        -0.183 |       15   |
|     18 | calibrated-ips (iic=False)                            |       57.2 |   0.017  |             0.4496 |          7.3 |       0.0777 |        -0.183 |       15   |
|     19 | orthogonalized-ips (iic=False)                        |       46   |   0.0925 |             0.6592 |          5.6 |       0.1653 |        -0.083 |       30   |
|     20 | raw-ips (iic=False)                                   |       43.2 |   0.092  |             0.6824 |          5.6 |       0.1739 |        -0.133 |       27.5 |
|     21 | raw-ips (iic=True)                                    |       43.1 |   0.092  |             0.6826 |          5.6 |       0.1741 |        -0.133 |       27.5 |
|     22 | tr-cpo (iic=False)                                    |       40.5 |   0.9022 |             0.338  |          9.9 |       0.0755 |        -0.017 |       42.5 |
|     23 | orthogonalized-ips (iic=True)                         |       38.6 |   0.0925 |             0.8178 |          5   |       0.2086 |        -0.083 |       30   |
|     24 | tr-cpo (iic=True)                                     |       38.3 |   0.9022 |             0.3939 |          8.8 |       0.0904 |        -0.017 |       42.5 |