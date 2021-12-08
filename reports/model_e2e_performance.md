# Model Performance

E2E performance of the final model selection candidates. This data is extracted from the
`fm_e2e_evaluate_xxx` notebooks.

## Mean Estimator
- missing prediction: 0.58%
- spurious prediction: 9.80%
- y_true and y_pred NaN: 1.88%
- r2 score: 0.539
- root mean squared error: 9.84
- mean absolute error: 4.89

## LGBM/9 - best
- missing prediction: 0.50%
- spurious prediction: 14.58%
- y_true and y_pred NaN: 0.00%
- r2 score: 0.622
- root mean squared error: 8.92
- mean absolute error: 4.91

## LGBM/11 - least features
- missing prediction: 0.50%
- spurious prediction: 14.58%
- y_true and y_pred NaN: 0.00%
- r2 score: 0.619
- root mean squared error: 8.95
- mean absolute error: 4.96

## LinearRegression/3

- root mean squared error: 11.08
- mean absolute error: 6.67