"""
Estimate BMI based on Gender, Height & Weight
-----------------------------------------------
Internship Project - Machine Learning Domain (June 2026)

This script:
1. Loads a Gender/Height/Weight/BMI dataset (or generates a realistic
   synthetic one if no file is supplied — replace DATA_PATH with your
   own CSV once you have real data).
2. Explores and visualizes the data.
3. Preprocesses features (encodes Gender).
4. Trains and compares regression models to predict BMI.
5. Evaluates performance (MAE, RMSE, R2).
6. Saves the trained model so it can be reused for new predictions.

Expected CSV columns if you use your own file:
    Gender, Height (in cm), Weight (in kg)
Optionally a BMI column — if absent, it is computed from the formula.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

RANDOM_STATE = 42
DATA_PATH = "bmi_data.csv"   # <-- point this at your real dataset if you have one
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# 1. Load or generate data
# ---------------------------------------------------------------------
def generate_synthetic_data(n=500, seed=RANDOM_STATE):
    """Generates a realistic Gender/Height/Weight dataset and computes BMI.
    Use this only until you have a real dataset to plug in."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n)

    # Height in cm, roughly normal per gender (realistic ranges)
    height_cm = np.where(
        gender == "Male",
        rng.normal(171, 7, n),
        rng.normal(160, 7, n),
    ).clip(140, 210)

    # Weight in kg, correlated with height + random variation
    base_weight = (height_cm - 100) * 0.9
    weight_kg = (base_weight + rng.normal(0, 8, n)).clip(35, 150)

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    df = pd.DataFrame({
        "Gender": gender,
        "Height": height_cm.round(1),
        "Weight": weight_kg.round(1),
        "BMI": bmi.round(2),
    })
    return df


if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded real dataset from {DATA_PATH}: {df.shape}")
else:
    print(f"No file found at '{DATA_PATH}' — using synthetic data instead.")
    print("Replace DATA_PATH with your real CSV when you have one.\n")
    df = generate_synthetic_data()
    df.to_csv(os.path.join(OUTPUT_DIR, "bmi_data.csv"), index=False)

# If BMI column missing, compute it from Height/Weight (assumes cm & kg)
if "BMI" not in df.columns:
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

print(df.head())
print(df.describe())


# ---------------------------------------------------------------------
# 2. Quick EDA (saved as images for your report)
# ---------------------------------------------------------------------
sns.set_style("whitegrid")

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="Height", y="Weight", hue="Gender", alpha=0.7)
plt.title("Height vs Weight by Gender")
plt.savefig(os.path.join(OUTPUT_DIR, "height_vs_weight.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="BMI", hue="Gender", kde=True, bins=25)
plt.title("BMI Distribution by Gender")
plt.savefig(os.path.join(OUTPUT_DIR, "bmi_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

corr = df.drop(columns=["Gender"]).corr(numeric_only=True)
plt.figure(figsize=(4, 3))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nEDA plots saved to '{OUTPUT_DIR}/'")


# ---------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------
le = LabelEncoder()
df["Gender_encoded"] = le.fit_transform(df["Gender"])  # Female=0, Male=1 (alphabetical)

X = df[["Gender_encoded", "Height", "Weight"]]
y = df["BMI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# ---------------------------------------------------------------------
# 4. Train & compare models
# ---------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
best_model_name, best_model, best_r2 = None, None, -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 4)})

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

print(f"\nBest model: {best_model_name} (R2 = {best_r2:.4f})")


# ---------------------------------------------------------------------
# 5. Predicted vs Actual plot for the best model
# ---------------------------------------------------------------------
best_preds = best_model.predict(X_test)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, best_preds, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title(f"Actual vs Predicted BMI ({best_model_name})")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------
# 6. Save the trained model + encoder for reuse
# ---------------------------------------------------------------------
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "bmi_model.pkl"))
joblib.dump(le, os.path.join(OUTPUT_DIR, "gender_encoder.pkl"))
print(f"\nModel saved to '{OUTPUT_DIR}/bmi_model.pkl'")


# ---------------------------------------------------------------------
# 7. Example: predict BMI for a new person
# ---------------------------------------------------------------------
def predict_bmi(gender: str, height_cm: float, weight_kg: float) -> float:
    """Predict BMI for a new person using the trained model."""
    gender_encoded = le.transform([gender])[0]
    input_df = pd.DataFrame([[gender_encoded, height_cm, weight_kg]],
                             columns=["Gender_encoded", "Height", "Weight"])
    return round(best_model.predict(input_df)[0], 2)


if __name__ == "__main__":
    example = predict_bmi("Male", 175, 70)
    print(f"\nExample prediction — Male, 175cm, 70kg -> Predicted BMI: {example}")
