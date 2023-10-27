# 永豐AI GO競賽-攻房戰

## Notice
main.ipynb, dataPreprocessing.ipynb, debugForUtils.ipynb 為舊版本，現在不使用。以使用LightGBMV*.ipynb 為主

## Model Performance (Submit MAPE)
1. **LightGBMV2.ipynb**: 8.492715
2. **LightGBMV23.ipynb**: 8.352278
3. **LightGBMV24.ipynb**: 8.369518
4. **LightGBMV3.ipynb**: 8.493608
5. **LightGBMV4.ipynb**: 8.374716
6. **XGBoostV1.ipynb**: 12.62969
-- can compare these model quickly by LOG.pptx--

## dataset
* public_dataset.csv: test data
* training_data.csv: training data
* public_submission_template.csv: submission_template (can ignore it)

## Current Goal
-[ ] Stack different models (like the p.15 in the PPT).
-[ ] Try to make or add more data to the model.
-[ ] The way using external data is calculate the number locate in the given radius, need to decide the appropriate radius, or decide the better way to use it.
