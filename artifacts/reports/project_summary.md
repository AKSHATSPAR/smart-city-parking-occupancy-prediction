# Smart Parking Project Summary

## Dataset Snapshot
- Records after cleaning: 18268
- Parking systems: 14
- Time span: 2016-10-04T08:00:00 to 2016-12-19T16:30:00
- Occupancy anomalies capped during cleaning: 241

## Model Outcome
- Time-based train/test split boundary: 2016-12-02 15:00:00
- Best model by RMSE: XGBoost
- RMSE: 0.0272
- MAE: 0.0191
- Utilization-band accuracy: 0.9193

## Clustering Insight
- Optimal cluster count: 5
- Silhouette score: 0.2902
- PCA explained variance: 0.6408

## Multi-Horizon Forecasting
- 30 minutes: RMSE 0.0164, MAE 0.0113, band accuracy 0.9522
- 1 hour: RMSE 0.0248, MAE 0.0175, band accuracy 0.9304
- 2 hours: RMSE 0.0378, MAE 0.0273, band accuracy 0.8831

## Recommendation Snapshot
- Top recommended parking system right now: BHMBCCMKT01
- Predicted free spaces in 1 hour: 359
- Risk band: Low