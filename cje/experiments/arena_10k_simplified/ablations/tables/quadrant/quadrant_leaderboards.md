# Quadrant-Specific Leaderboards

## Small-Low Quadrant

|   Rank | Estimator                       |   AggScore |   RMSE_d |   IntervalScore_OA |   CalibScore |   SE_GeoMean |   Kendall_tau |   Top1_Acc |
|-------:|:--------------------------------|-----------:|---------:|-------------------:|-------------:|-------------:|--------------:|-----------:|
|      1 | stacked-dr (iic=True)           |      100   |   0.0081 |             0.0573 |          5   |       0.0146 |         0.667 |        nan |
|      2 | stacked-dr (iic=False)          |      100   |   0.008  |             0.0573 |          5   |       0.0146 |         0.667 |        nan |
|      3 | dr-cpo (calib=True, iic=True)   |       89.1 |   0.0083 |             0.1918 |          5   |       0.0489 |         0.5   |        nan |
|      4 | dr-cpo (calib=True, iic=False)  |       89.1 |   0.0083 |             0.1924 |          5   |       0.0491 |         0.5   |        nan |
|      5 | oc-dr-cpo (iic=True)            |       87.8 |   0.0399 |             0.092  |          5   |       0.0235 |         0.333 |        nan |
|      6 | oc-dr-cpo (iic=False)           |       87.6 |   0.0399 |             0.0957 |          5   |       0.0244 |         0.333 |        nan |
|      7 | dr-cpo (calib=False, iic=False) |       85.6 |   0.0402 |             0.2629 |          5   |       0.0671 |         0.5   |        nan |
|      8 | dr-cpo (calib=False, iic=True)  |       85.5 |   0.0402 |             0.2659 |          5   |       0.0678 |         0.5   |        nan |
|      9 | tr-cpo-e (iic=False)            |       70.5 |   0.2066 |             0.3147 |          5   |       0.0803 |         0.167 |        nan |
|     10 | tr-cpo-e (iic=True)             |       70.1 |   0.2066 |             0.3228 |          5   |       0.0824 |         0.167 |        nan |
|     11 | calibrated-ips (iic=True)       |       66.2 |   0.0112 |             0.4519 |          5   |       0.0888 |         0     |        nan |
|     12 | calibrated-ips (iic=False)      |       66.2 |   0.0112 |             0.4519 |          5   |       0.0888 |         0     |        nan |
|     13 | orthogonalized-ips (iic=False)  |       43   |   0.1267 |             0.8678 |          5   |       0.2214 |         0     |        nan |
|     14 | orthogonalized-ips (iic=True)   |       42.9 |   0.1267 |             0.8693 |          5   |       0.2218 |         0     |        nan |
|     15 | raw-ips (iic=False)             |       42.6 |   0.1257 |             0.8768 |          5   |       0.2237 |         0     |        nan |
|     16 | raw-ips (iic=True)              |       42.2 |   0.1257 |             0.888  |          5   |       0.2265 |         0     |        nan |
|     17 | tr-cpo (iic=False)              |       14.8 |   1.7731 |             1.3633 |         19.2 |       0.1564 |         0.167 |        nan |
|     18 | tr-cpo (iic=True)               |       14   |   1.7731 |             1.3911 |         19.2 |       0.1599 |         0.167 |        nan |

## Small-High Quadrant

|   Rank | Estimator                       |   AggScore |   RMSE_d |   IntervalScore_OA |   CalibScore |   SE_GeoMean |   Kendall_tau |   Top1_Acc |
|-------:|:--------------------------------|-----------:|---------:|-------------------:|-------------:|-------------:|--------------:|-----------:|
|      1 | dr-cpo (calib=True, iic=False)  |       96.2 |   0.0094 |             0.0724 |          5   |       0.0185 |         0.889 |        nan |
|      2 | dr-cpo (calib=True, iic=True)   |       96.2 |   0.0094 |             0.0728 |          5   |       0.0186 |         0.889 |        nan |
|      3 | oc-dr-cpo (iic=False)           |       95.1 |   0.0094 |             0.0352 |         23.9 |       0.0066 |         1     |        nan |
|      4 | oc-dr-cpo (iic=True)            |       91.4 |   0.0094 |             0.039  |         38.9 |       0.0056 |         1     |        nan |
|      5 | tr-cpo-e (iic=True)             |       90.4 |   0.0254 |             0.063  |          5   |       0.0161 |         0.556 |        nan |
|      6 | tr-cpo-e (iic=False)            |       90.4 |   0.0254 |             0.0642 |          5   |       0.0164 |         0.556 |        nan |
|      7 | stacked-dr (iic=False)          |       87.9 |   0.0112 |             0.0677 |         50   |       0.0072 |         1     |        nan |
|      8 | stacked-dr (iic=True)           |       87.9 |   0.0119 |             0.0673 |         50   |       0.0072 |         1     |        nan |
|      9 | dr-cpo (calib=False, iic=False) |       71.8 |   0.0642 |             0.1556 |          5   |       0.0397 |        -0.222 |        nan |
|     10 | dr-cpo (calib=False, iic=True)  |       71.6 |   0.0642 |             0.159  |          5   |       0.0406 |        -0.222 |        nan |
|     11 | calibrated-ips (iic=True)       |       66.2 |   0.0155 |             0.44   |          5   |       0.0796 |        -0.222 |        nan |
|     12 | calibrated-ips (iic=False)      |       66.2 |   0.0155 |             0.44   |          5   |       0.0796 |        -0.222 |        nan |
|     13 | orthogonalized-ips (iic=False)  |       39.5 |   0.1026 |             0.8865 |          5   |       0.2262 |         0     |        nan |
|     14 | orthogonalized-ips (iic=True)   |       39.5 |   0.1026 |             0.8874 |          5   |       0.2264 |         0     |        nan |
|     15 | raw-ips (iic=False)             |       37.7 |   0.1348 |             0.9152 |          8.9 |       0.1999 |        -0.111 |        nan |
|     16 | raw-ips (iic=True)              |       36.9 |   0.1348 |             0.9355 |          8.9 |       0.2044 |        -0.111 |        nan |
|     17 | tr-cpo (iic=False)              |       36.2 |   0.4475 |             0.3557 |         12.8 |       0.0753 |        -0.333 |        nan |
|     18 | tr-cpo (iic=True)               |       36   |   0.4475 |             0.3644 |         12.8 |       0.0753 |        -0.333 |        nan |

## Large-Low Quadrant

|   Rank | Estimator                       |   AggScore |   RMSE_d |   IntervalScore_OA |   CalibScore |   SE_GeoMean |   Kendall_tau |   Top1_Acc |
|-------:|:--------------------------------|-----------:|---------:|-------------------:|-------------:|-------------:|--------------:|-----------:|
|      1 | dr-cpo (calib=True, iic=True)   |       96.1 |   0.0176 |             0.0954 |          5   |       0.0243 |         1     |        nan |
|      2 | dr-cpo (calib=True, iic=False)  |       96.1 |   0.0176 |             0.0955 |          5   |       0.0244 |         1     |        nan |
|      3 | dr-cpo (calib=False, iic=True)  |       88.9 |   0.018  |             0.1667 |          5   |       0.0425 |         0.833 |        nan |
|      4 | dr-cpo (calib=False, iic=False) |       88.9 |   0.018  |             0.1668 |          5   |       0.0426 |         0.833 |        nan |
|      5 | oc-dr-cpo (iic=False)           |       86.7 |   0.0179 |             0.1967 |         86.7 |       0.0059 |         1     |        nan |
|      6 | oc-dr-cpo (iic=True)            |       86.5 |   0.0179 |             0.2119 |         86.7 |       0.0057 |         1     |        nan |
|      7 | tr-cpo-e (iic=True)             |       81.7 |   0.048  |             0.1312 |          5   |       0.0335 |         0.333 |        nan |
|      8 | tr-cpo-e (iic=False)            |       81.7 |   0.048  |             0.1314 |          5   |       0.0335 |         0.333 |        nan |
|      9 | tr-cpo (iic=True)               |       77.7 |   0.0407 |             0.1411 |         10.8 |       0.0324 |         0     |        nan |
|     10 | tr-cpo (iic=False)              |       77.7 |   0.0407 |             0.1408 |         10.8 |       0.0325 |         0     |        nan |
|     11 | calibrated-ips (iic=False)      |       52.1 |   0.0255 |             0.5757 |         10.8 |       0.0966 |        -0.5   |        nan |
|     12 | calibrated-ips (iic=True)       |       52.1 |   0.0255 |             0.5757 |         10.8 |       0.0966 |        -0.5   |        nan |
|     13 | orthogonalized-ips (iic=True)   |       14.5 |   0.2507 |             0.7159 |         16.7 |       0.1196 |        -0.667 |        nan |
|     14 | orthogonalized-ips (iic=False)  |       14.4 |   0.2507 |             0.7174 |         16.7 |       0.1198 |        -0.667 |        nan |
|     15 | raw-ips (iic=False)             |       12.5 |   0.2517 |             0.6162 |         10.8 |       0.1414 |        -0.667 |        nan |
|     16 | raw-ips (iic=True)              |       12.4 |   0.2517 |             0.6153 |         10.8 |       0.1417 |        -0.667 |        nan |

## Large-High Quadrant

|   Rank | Estimator                       |   AggScore |   RMSE_d |   IntervalScore_OA |   CalibScore |   SE_GeoMean |   Kendall_tau |   Top1_Acc |
|-------:|:--------------------------------|-----------:|---------:|-------------------:|-------------:|-------------:|--------------:|-----------:|
|      1 | oc-dr-cpo (iic=False)           |       99.9 |   0.0028 |             0.0127 |          5   |       0.0032 |         1     |        nan |
|      2 | dr-cpo (calib=True, iic=False)  |       97.6 |   0.003  |             0.0455 |          5   |       0.0116 |         1     |        nan |
|      3 | dr-cpo (calib=True, iic=True)   |       97.6 |   0.003  |             0.0455 |          5   |       0.0116 |         1     |        nan |
|      4 | tr-cpo-e (iic=True)             |       97.3 |   0.0034 |             0.0299 |          5   |       0.0076 |         0.889 |        nan |
|      5 | tr-cpo-e (iic=False)            |       97.2 |   0.0034 |             0.0306 |          5   |       0.0078 |         0.889 |        nan |
|      6 | oc-dr-cpo (iic=True)            |       88.6 |   0.0028 |             0.0123 |          8.9 |       0.0029 |         1     |        nan |
|      7 | dr-cpo (calib=False, iic=False) |       82.2 |   0.0208 |             0.0942 |          5   |       0.024  |         0.222 |        nan |
|      8 | dr-cpo (calib=False, iic=True)  |       82.2 |   0.0208 |             0.0953 |          5   |       0.0243 |         0.222 |        nan |
|      9 | tr-cpo (iic=True)               |       71.2 |   0.0502 |             0.0976 |          5   |       0.0249 |        -0.333 |        nan |
|     10 | tr-cpo (iic=False)              |       71.2 |   0.0502 |             0.098  |          5   |       0.025  |        -0.333 |        nan |
|     11 | calibrated-ips (iic=False)      |       48.9 |   0.0152 |             0.5369 |          5   |       0.0971 |        -0.667 |        nan |
|     12 | calibrated-ips (iic=True)       |       48.9 |   0.0152 |             0.5369 |          5   |       0.0971 |        -0.667 |        nan |
|     13 | orthogonalized-ips (iic=True)   |       20.1 |   0.1951 |             0.5482 |          5   |       0.1399 |        -0.333 |        nan |
|     14 | orthogonalized-ips (iic=False)  |       20   |   0.1951 |             0.5489 |          5   |       0.14   |        -0.333 |        nan |
|     15 | raw-ips (iic=False)             |       15.5 |   0.2132 |             0.5784 |          5   |       0.1476 |        -0.333 |        nan |
|     16 | raw-ips (iic=True)              |       15.5 |   0.2132 |             0.5797 |          5   |       0.1479 |        -0.333 |        nan |

