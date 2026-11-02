from ucimlrepo import fetch_ucirepo
import pandas as pd
import os


def main():

    print("Downloading UCI Heart Disease dataset...")

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    print("Feature columns:")
    print(X.columns.tolist())

    print("\nTarget columns:")
    print(y.columns.tolist())

    df = pd.concat([X, y], axis=1)

    # Rename the 14 columns used in this project
    df.columns = [
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
        "thal",
        "target"
    ]

    # Convert values to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert original UCI target:
    # 0 -> 0
    # 1,2,3,4 -> 1
    df["target"] = df["target"].apply(
        lambda value: 1 if value > 0 else 0
    )

    os.makedirs("dataset", exist_ok=True)

    output_file = "dataset/heart.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\nDataset saved successfully.")
    print("File:", output_file)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nTarget distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()