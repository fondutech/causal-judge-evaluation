## Table A5: Calibration Boundary Analysis

| Estimator                       |   Mean_Dist_Boundary |   Min_Dist_Boundary |   Pct_Near_Boundary |   Unhelpful_Mean_Dist |   Unhelpful_Min_Dist |   Outlier_Rate | Support   |
|:--------------------------------|---------------------:|--------------------:|--------------------:|----------------------:|---------------------:|---------------:|:----------|
| orthogonalized-ips (iic=True)   |            0.152489  |           0.0335626 |            40       |             0.21476   |            0.117409  |             10 | Weak      |
| orthogonalized-ips (iic=False)  |            0.152489  |           0.0335626 |            40       |             0.21476   |            0.117409  |             10 | Weak      |
| raw-ips (iic=True)              |            0.151348  |          -0.039617  |            36.6667  |             0.222917  |            0.116225  |             10 | Weak      |
| raw-ips (iic=False)             |            0.151348  |          -0.039617  |            36.6667  |             0.222917  |            0.116225  |             10 | Weak      |
| tr-cpo (iic=False)              |           -0.0475401 |          -5.41159   |            15       |             0.0579017 |           -0.0679304 |             90 | Weak      |
| tr-cpo (iic=True)               |           -0.0475401 |          -5.41159   |            15       |             0.0579017 |           -0.0679304 |             90 | Weak      |
| tr-cpo-e (iic=False)            |            0.174352  |          -0.377043  |             8.33333 |             0.0619961 |           -0.22731   |             80 | Weak      |
| tr-cpo-e (iic=True)             |            0.174352  |          -0.377043  |             8.33333 |             0.0619961 |           -0.22731   |             80 | Weak      |
| oc-dr-cpo (iic=True)            |            0.192534  |           0.0418366 |             1.66667 |             0.0954904 |            0.0244898 |             90 | Weak      |
| oc-dr-cpo (iic=False)           |            0.192534  |           0.0418366 |             1.66667 |             0.0954904 |            0.0244898 |             90 | Weak      |
| dr-cpo (calib=False, iic=False) |            0.202587  |           0.0407269 |             1.66667 |             0.086492  |           -0.0434808 |             90 | Weak      |
| dr-cpo (calib=False, iic=True)  |            0.202587  |           0.0407269 |             1.66667 |             0.086492  |           -0.0434808 |             90 | Weak      |
| dr-cpo (calib=True, iic=False)  |            0.193363  |           0.143286  |             0       |             0.0934421 |            0.026044  |             90 | Weak      |
| dr-cpo (calib=True, iic=True)   |            0.193363  |           0.143286  |             0       |             0.0934421 |            0.026044  |             90 | Weak      |
| calibrated-ips (iic=False)      |            0.197264  |           0.137686  |             0       |             0.197329  |            0.159527  |              0 | Weak      |
| calibrated-ips (iic=True)       |            0.197264  |           0.137686  |             0       |             0.197329  |            0.159527  |              0 | Weak      |
| stacked-dr (iic=True)           |            0.193032  |           0.143212  |             0       |             0.0980917 |            0.0285161 |             85 | Weak      |
| stacked-dr (iic=False)          |            0.19322   |           0.143201  |             0       |             0.0960934 |            0.0285113 |             85 | Weak      |