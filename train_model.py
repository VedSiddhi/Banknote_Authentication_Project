
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")
STUDENT_DIR = os.path.join(BASE, "student_regression")

os.makedirs(MODELS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def save_comparison_chart(df, x_col, y_cols, title, filename):
    ax = df.set_index(x_col)[y_cols].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, filename), dpi=150)
    plt.close()


# ==================================================
# PART A: BANKNOTE CLASSIFICATION
# ==================================================

def train_classification():
    path = os.path.join(BASE, "data_banknote_authentication.txt")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing data_banknote_authentication.txt in project root."
        )

    columns = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(path, header=None, names=columns)

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates()
    df = df[df["class"].isin([0, 1])]

    X = df[columns[:-1]]
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            random_state=42, n_jobs=-1
        )
    }

    rows = []

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, pred, zero_division=0
            )
        })

        filename = (
            "decision_tree_classifier.pkl"
            if name == "Decision Tree"
            else "random_forest_classifier.pkl"
        )
        joblib.dump(model, os.path.join(MODELS, filename))

        print(f"\nClassification: {name}")
        print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
        print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_test, pred, zero_division=0):.4f}")
        print(f"F1 score: {f1_score(y_test, pred, zero_division=0):.4f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        os.path.join(RESULTS, "classification_metrics.csv"),
        index=False
    )
    save_comparison_chart(
        metrics, "Model",
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Banknote Classification Model Comparison",
        "classification_comparison.png"
    )

    print("\nBanknote classification training completed.")


# ==================================================
# PART B: STUDENT PERFORMANCE REGRESSION
# ==================================================

def train_regression():
    path = os.path.join(STUDENT_DIR, "student-mat.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing student_regression/student-mat.csv. "
            "Download and extract the UCI Student Performance dataset."
        )

    # UCI student CSV is semicolon-separated.
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    if "G3" not in df.columns:
        raise ValueError("Student dataset must contain target column G3.")

    # Exclude G1 and G2 to avoid using earlier grades to predict G3.
    feature_columns = [
        "age", "Medu", "Fedu", "studytime", "failures",
        "absences", "schoolsup", "famsup", "paid",
        "activities", "higher", "internet", "goout", "health"
    ]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing student dataset columns: {missing}")


    X = df[feature_columns].copy()
    y = pd.to_numeric(df["G3"], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns if col not in numeric_cols
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_cols),
        ("categorical", categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    regressors = {
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=8,
            random_state=42, n_jobs=-1
        )
    }

    rows = []

    for name, regressor in regressors.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)

        rows.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        filename = (
            "decision_tree_regressor.pkl"
            if name == "Decision Tree Regressor"
            else "random_forest_regressor.pkl"
        )
        joblib.dump(pipeline, os.path.join(MODELS, filename))

        print(f"\nRegression: {name}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        os.path.join(RESULTS, "regression_metrics.csv"),
        index=False
    )

    # MAE/RMSE are error measures (lower is better).
    ax = metrics.set_index("Model")[["MAE", "RMSE"]].plot(
        kind="bar", figsize=(9, 5)
    )
    ax.set_title("Student Regression Error Comparison")
    ax.set_ylabel("Error in grade points")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS, "regression_comparison.png"), dpi=150
    )
    plt.close()

    print("\nStudent regression training completed.")


# ==================================================
# RUN BOTH EXPERIMENTS
# ==================================================

if __name__ == "__main__":
    print("=" * 55)
    print("BANKNOTE CLASSIFICATION + STUDENT REGRESSION")
    print("=" * 55)

    train_classification()
    train_regression()

    print("\nALL FOUR MODELS TRAINED SUCCESSFULLY.")
    print("Models saved in:", MODELS)
    print("Evaluation results saved in:", RESULTS)
