from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

# ===== 1. LOAD DATA =====
# Updated path and target logic for Wine Quality dataset
# Replace with your actual local file path: r"E:\Beginner ML projects datasets\WineQT.csv"
df = pd.read_csv(r"E:\Beginner ML projects datasets\WineQT.csv")

# Clean specific WineQT missing values if any exist (safety check)
numerical_features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

for col in numerical_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# Handle duplicates commonly found in chemical samples
df = df.drop_duplicates()

# Create binary classification target (1 = Good Wine >= 6, 0 = Bad Wine < 6)
df["good_quality"] = np.where(df["quality"] >= 6, 1, 0)

# ===== 2. SPLIT DATA =====
# Drop non-predictive text/numeric IDs and raw quality scores
X = df.drop(columns=["Id", "quality", "good_quality"], errors="ignore")
y = df["good_quality"].astype(int)

# Since all features are continuous numeric metrics, ColumnTransformer acts as a passthrough pipeline
# This structure preserves your original pipeline layout in case you add categorical columns later
preprocessor = ColumnTransformer(
    transformers=[("num", "passthrough", X.columns)],
    remainder="drop",
)

# Fit and transform the data matrix
X_processed = preprocessor.fit_transform(X)

# Split the processed data using stratify to preserve the good/bad wine balance
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale continuous chemical metrics to prevent high-value features (like sulfur) from dominating
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 3. MODELS + TUNING =====
rf_estimators = [100, 200, 300]
rf_splits = [2, 5, 10]
xgb_estimators = [100, 200]
xgb_depths = [3, 5, 7]

models = {
    "Logistic": {
        "model": LogisticRegression(max_iter=1000), 
        "params": {}
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": rf_estimators,
            "max_depth": [5, 10, None],
            "min_samples_split": rf_splits
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
        "params": {
            "n_estimators": xgb_estimators,
            "max_depth": xgb_depths,
            "learning_rate": [0.01, 0.1, 0.3]
        }
    }
}

results = {}

print("Training wine quality classification models... grab chai ☕")
print("-" * 40)

for name, config in models.items():
    model = config["model"]
    params = config["params"]

    if params:
        search = RandomizedSearchCV(
            model, params, n_iter=10, cv=3, scoring="accuracy", random_state=42, n_jobs=-1
        )
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        print(f"{name}: Tuning done. Best params = {search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
        print(f"{name}: Trained")

    # Evaluate performance
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "F1_Score": f1, "Model": best_model}

# ===== 4. SHOW WINNER =====
print("\n" + "=" * 40)
print("FINAL RESULTS")
print("=" * 40)

for name, scores in sorted(results.items(), key=lambda x: x[1]["Accuracy"], reverse=True):
    print(f"{name:15} | Accuracy = {scores['Accuracy']:.4f} | F1-Score = {scores['F1_Score']:.4f}")

winner_name = max(results, key=lambda x: results[x]["Accuracy"])
print("\n🏆 WINNER:", winner_name)
print(f"Best Accuracy: {results[winner_name]['Accuracy']:.4f}")
print(f"Use this model: results['{winner_name}']['Model']")

# Save artifacts tailored for Wine Quality prediction
artifacts = {
    "model": results[winner_name]["Model"],
    "preprocessor": preprocessor,
    "scaler": scaler,
}

with open("WineQualityPrediction.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nModel, preprocessor, and scaler saved cleanly to WineQualityPrediction.pkl")
