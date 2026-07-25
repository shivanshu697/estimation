# BMI Estimation using Machine Learning

Machine Learning Internship Project — Naviotech Solution Pvt. Ltd. (June 2026)

## Overview

This project estimates a person's **Body Mass Index (BMI)** using three simple inputs — **Gender, Height, and Weight** — framed as a supervised regression problem. Two models were trained and compared: **Linear Regression** and **Random Forest Regressor**.

## Project Structure

| File | Description |
|---|---|
| `bmi_estimation.py` | Main script — loads data, runs EDA, trains models, evaluates, and saves the model |
| `bmi_data.csv` | Dataset used (Gender, Height, Weight, BMI) |
| `bmi_model.pkl` | Final trained model (Linear Regression), saved with joblib |
| `model_comparison.csv` | Performance metrics (MAE, RMSE, R²) for both models |
| `height_vs_weight.png` | EDA chart — Height vs Weight by Gender |
| `bmi_distribution.png` | EDA chart — BMI distribution by Gender |
| `correlation_heatmap.png` | EDA chart — feature correlation heatmap |
| `actual_vs_predicted.png` | Model evaluation chart — predicted vs actual BMI |

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```
2. Run the script:
   ```bash
   python bmi_estimation.py
   ```
   This trains both models, prints evaluation metrics, regenerates all charts, and saves the trained model.

## Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Linear Regression** | 0.267 | 0.369 | **0.9834** |
| Random Forest Regressor | 0.261 | 0.397 | 0.9808 |

Linear Regression was selected as the final model — it achieved the highest R² score, and since BMI is mathematically close to a linear function of Height and Weight, it also offers the best interpretability.

## Using the Trained Model

```python
import joblib

model = joblib.load("bmi_model.pkl")
# Gender_encoded: Female = 0, Male = 1
predicted_bmi = model.predict([[1, 175, 70]])  # Male, 175cm, 70kg
print(predicted_bmi)
```

## Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- joblib

## Author

Shivanshu
Machine Learning Internship — Naviotech Solution Pvt. Ltd.
