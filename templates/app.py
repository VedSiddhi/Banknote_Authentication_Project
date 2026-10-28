
import os
import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

FEATURES = ["variance", "skewness", "curtosis", "entropy"]

STUDENT_FEATURES = [
    "age", "Medu", "Fedu", "studytime", "failures",
    "absences", "schoolsup", "famsup", "paid",
    "activities", "higher", "internet", "goout", "health"
]

CLASSIFIER_FILES = {
    "Decision Tree": "decision_tree_classifier.pkl",
    "Random Forest": "random_forest_classifier.pkl"
}

REGRESSOR_FILES = {
    "Decision Tree Regressor": "decision_tree_regressor.pkl",
    "Random Forest Regressor": "random_forest_regressor.pkl"
}

models = {}
load_errors = []

for name, filename in {
    **CLASSIFIER_FILES,
    **REGRESSOR_FILES
}.items():
    path = os.path.join(BASE_DIR, "models", filename)

    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{filename} is missing. Run train_model.py first."
            )
        models[name] = joblib.load(path)
        print(f"Loaded: {name}")
    except Exception as exc:
        load_errors.append(f"{name}: {exc}")
        print(f"Could not load {name}: {exc}")


STUDENT_OPTIONS = {
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"]
}


def default_student_values():
    return {
        "age": "17",
        "Medu": "2",
        "Fedu": "2",
        "studytime": "2",
        "failures": "0",
        "absences": "0",
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "higher": "yes",
        "internet": "yes",
        "goout": "3",
        "health": "3"
    }


@app.route("/", methods=["GET", "POST"])
def home():
    task = request.form.get("task", "classification")
    if task not in ("classification", "regression"):
        task = "classification"

    banknote_values = {key: "" for key in FEATURES}
    student_values = default_student_values()

    selected_classifier = request.form.get(
        "classifier", "Random Forest"
    )
    selected_regressor = request.form.get(
        "regressor", "Random Forest Regressor"
    )

    prediction = None
    regression_result = None
    error = None

    if request.method == "POST":
        try:
            if task == "classification":
                for feature in FEATURES:
                    banknote_values[feature] = request.form.get(
                        feature, ""
                    ).strip()

                if selected_classifier not in CLASSIFIER_FILES:
                    raise ValueError("Select a valid classifier.")

                if selected_classifier not in models:
                    raise RuntimeError(
                        f"{selected_classifier} is unavailable. "
                        "Check model files and run training."
                    )

                try:
                    numbers = [
                        float(banknote_values[f]) for f in FEATURES
                    ]
                except (ValueError, TypeError):
                    raise ValueError(
                        "Enter valid numbers for all four banknote features."
                    )

                if not np.isfinite(numbers).all():
                    raise ValueError("Feature values must be finite numbers.")

                model = models[selected_classifier]
                sample = np.array([numbers], dtype=float)
                class_number = int(model.predict(sample)[0])

                confidence = None
                if hasattr(model, "predict_proba"):
                    confidence = round(
                        float(np.max(model.predict_proba(sample)[0])) * 100,
                        2
                    )

                prediction = {
                    "label": f"Class {class_number}",
                    "class_number": class_number,
                    "confidence": confidence,
                    "model": selected_classifier
                }

            else:
                student_values = default_student_values()
                for feature in STUDENT_FEATURES:
                    student_values[feature] = request.form.get(
                        feature, student_values[feature]
                    ).strip()

                if selected_regressor not in REGRESSOR_FILES:
                    raise ValueError("Select a valid regression model.")

                if selected_regressor not in models:
                    raise RuntimeError(
                        f"{selected_regressor} is unavailable. "
                        "Check model files and run training."
                    )

                numeric_features = [
                    "age", "Medu", "Fedu", "studytime",
                    "failures", "absences", "goout", "health"
                ]

                row = {}
                for feature in numeric_features:
                    try:
                        row[feature] = float(student_values[feature])
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Enter a valid number for {feature}."
                        )

                for feature in STUDENT_OPTIONS:
                    value = student_values[feature].lower()
                    if value not in STUDENT_OPTIONS[feature]:
                        raise ValueError(
                            f"Invalid selection for {feature}."
                        )
                    row[feature] = value

                if not np.isfinite(
                    [row[f] for f in numeric_features]
                ).all():
                    raise ValueError("Numeric inputs must be finite.")

                if not 15 <= row["age"] <= 22:
                    raise ValueError("Age must be between 15 and 22.")
                if not 0 <= row["Medu"] <= 4 or not 0 <= row["Fedu"] <= 4:
                    raise ValueError("Parent education values must be 0–4.")
                if not 1 <= row["studytime"] <= 4:
                    raise ValueError("Study time must be between 1 and 4.")
                if not 0 <= row["failures"] <= 4:
                    raise ValueError("Failures must be between 0 and 4.")
                if row["absences"] < 0:
                    raise ValueError("Absences cannot be negative.")
                if not 1 <= row["goout"] <= 5 or not 1 <= row["health"] <= 5:
                    raise ValueError("Go-out and health ratings must be 1–5.")

                import pandas as pd
                sample = pd.DataFrame(
                    [[row[f] for f in STUDENT_FEATURES]],
                    columns=STUDENT_FEATURES
                )

                predicted_grade = float(
                    models[selected_regressor].predict(sample)[0]
                )
                predicted_grade = max(0.0, min(20.0, predicted_grade))

                regression_result = {
                    "grade": round(predicted_grade, 2),
                    "model": selected_regressor
                }

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        task=task,
        banknote_values=banknote_values,
        student_values=student_values,
        prediction=prediction,
        regression_result=regression_result,
        selected_classifier=selected_classifier,
        selected_regressor=selected_regressor,
        student_options=STUDENT_OPTIONS,
        error=error
    )


if __name__ == "__main__":
    print("Project directory:", BASE_DIR)
    print(
        "Template exists:",
        os.path.isfile(
            os.path.join(BASE_DIR, "templates", "index.html")
        )
    )
    app.run(debug=True)
