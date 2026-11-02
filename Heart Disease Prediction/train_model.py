import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


DATASET_PATH = "dataset/heart.csv"
MODEL_DIR = "model"


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor


def build_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=7
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                random_state=42
            ),

        "SVM":
            SVC(
                probability=True,
                kernel="rbf",
                random_state=42
            )
    }


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = df.replace(
        ["?", "NA", "N/A", ""],
        np.nan
    )

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["target"] = pd.to_numeric(
        df["target"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["target"]
    )

    X = df[FEATURE_COLUMNS]

    y = df["target"].astype(int)

    print("Dataset shape:", df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = build_models()

    trained_models = {}
    metrics = {}

    best_model_name = None
    best_model = None
    best_accuracy = -1

    print("\nTraining models...\n")

    for name, classifier in models.items():

        pipeline = Pipeline([
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                classifier
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_test,
            predictions
        ).tolist()

        trained_models[name] = pipeline

        metrics[name] = {
            "accuracy": round(
                accuracy * 100,
                2
            ),
            "precision": round(
                precision * 100,
                2
            ),
            "recall": round(
                recall * 100,
                2
            ),
            "f1_score": round(
                f1 * 100,
                2
            ),
            "confusion_matrix": matrix
        }

        print(
            f"{name}: "
            f"Accuracy = {accuracy * 100:.2f}%"
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model_name = name
            best_model = pipeline

    bundle = {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "models": trained_models,
        "feature_columns": FEATURE_COLUMNS
    }

    joblib.dump(
        bundle,
        os.path.join(
            MODEL_DIR,
            "model_bundle.pkl"
        )
    )

    metrics_data = {
        "best_model": best_model_name,
        "models": metrics
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )

    print("\n--------------------------------")
    print("Training completed!")
    print("--------------------------------")

    print(
        "Best Model:",
        best_model_name
    )

    print(
        "Best Accuracy:",
        round(
            best_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "\nSaved:",
        "model/model_bundle.pkl"
    )

    print(
        "Saved:",
        "model/metrics.json"
    )


if __name__ == "__main__":
    main()