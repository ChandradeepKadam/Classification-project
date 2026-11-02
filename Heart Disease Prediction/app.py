import csv
import io
import json
import os
import sqlite3
from datetime import datetime


import joblib
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    redirect,
    url_for
)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


app = Flask(__name__)


MODEL_PATH = "model/model_bundle.pkl"
METRICS_PATH = "model/metrics.json"

DATABASE_PATH = "database/predictions.db"
REPORT_DIRECTORY = "reports"


os.makedirs(
    "database",
    exist_ok=True
)

os.makedirs(
    REPORT_DIRECTORY,
    exist_ok=True
)


# ---------------------------------------
# Load Model
# ---------------------------------------

bundle = joblib.load(
    MODEL_PATH
)

best_model = bundle["best_model"]

best_model_name = bundle["best_model_name"]

feature_columns = bundle["feature_columns"]


# ---------------------------------------
# Load Metrics
# ---------------------------------------

with open(
    METRICS_PATH,
    "r",
    encoding="utf-8"
) as file:

    metrics_data = json.load(file)


# ---------------------------------------
# Database
# ---------------------------------------

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca REAL,
            thal REAL,

            prediction INTEGER,

            result TEXT,

            probability REAL,

            model_name TEXT,

            created_at TEXT

        )
    """)

    connection.commit()

    connection.close()


init_database()


# ---------------------------------------
# Helper
# ---------------------------------------

def create_input_data(form_data):

    data = {
        "age": float(form_data["age"]),
        "sex": int(form_data["sex"]),
        "cp": int(form_data["cp"]),
        "trestbps": float(form_data["trestbps"]),
        "chol": float(form_data["chol"]),
        "fbs": int(form_data["fbs"]),
        "restecg": int(form_data["restecg"]),
        "thalach": float(form_data["thalach"]),
        "exang": int(form_data["exang"]),
        "oldpeak": float(form_data["oldpeak"]),
        "slope": int(form_data["slope"]),
        "ca": float(form_data["ca"]),
        "thal": float(form_data["thal"])
    }

    return pd.DataFrame(
        [data],
        columns=feature_columns
    )

def make_prediction(input_data):

    prediction = best_model.predict(
        input_data
    )[0]

    probabilities = best_model.predict_proba(
        input_data
    )[0]

    probability = probabilities[1] * 100

    if prediction == 1:

        result = "Heart Disease Detected"

    else:

        result = "No Heart Disease Detected"

    return (
        int(prediction),
        result,
        round(probability, 2)
    )


def save_prediction(
    values,
    prediction,
    result,
    probability
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (

            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
            prediction,
            result,
            probability,
            model_name,
            created_at

        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?
        )
    """, (

        values["age"],
        values["sex"],
        values["cp"],
        values["trestbps"],
        values["chol"],
        values["fbs"],
        values["restecg"],
        values["thalach"],
        values["exang"],
        values["oldpeak"],
        values["slope"],
        values["ca"],
        values["thal"],
        prediction,
        result,
        probability,
        best_model_name,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    prediction_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return prediction_id


# ---------------------------------------
# Home
# ---------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        model_name=best_model_name
    )


# ---------------------------------------
# Prediction
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("\n========== PREDICTION START ==========")

        print("Form data:")
        print(request.form)

        # Create input DataFrame
        input_data = create_input_data(
            request.form
        )

        print("\nInput DataFrame:")
        print(input_data)

        print("\nColumns:")
        print(input_data.columns.tolist())

        # Make prediction
        prediction = best_model.predict(
            input_data
        )[0]

        print("\nPrediction:")
        print(prediction)

        # Get probability
        probability_values = best_model.predict_proba(
            input_data
        )[0]

        probability = probability_values[1] * 100

        print("\nProbability:")
        print(probability)

        # Result
        if int(prediction) == 1:

            result = "Heart Disease Detected"

        else:

            result = "No Heart Disease Detected"

        # Convert input to dictionary
        values = input_data.iloc[0].to_dict()

        # Save prediction
        prediction_id = save_prediction(
            values,
            int(prediction),
            result,
            round(probability, 2)
        )

        print("\nPrediction ID:")
        print(prediction_id)

        print("========== PREDICTION SUCCESS ==========\n")

        return render_template(
            "result.html",
            prediction_id=prediction_id,
            result=result,
            probability=round(probability, 2),
            model_name=best_model_name
        )

    except Exception as error:

        print("\n========== PREDICTION ERROR ==========")
        print(type(error).__name__)
        print(str(error))
        print("=======================================\n")

        return render_template(
            "result.html",
            result="Prediction Error",
            probability=0,
            model_name=best_model_name,
            error=f"{type(error).__name__}: {str(error)}"
        )


# ---------------------------------------
# Dashboard
# ---------------------------------------

@app.route("/dashboard")
def dashboard():

    total_predictions = 0
    disease_predictions = 0
    no_disease_predictions = 0

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) AS total FROM predictions"
    )

    total_predictions = cursor.fetchone()["total"]

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE prediction = 1
        """
    )

    disease_predictions = cursor.fetchone()["total"]

    no_disease_predictions = (
        total_predictions -
        disease_predictions
    )

    connection.close()

    return render_template(
        "dashboard.html",

        total_predictions=total_predictions,

        disease_predictions=disease_predictions,

        no_disease_predictions=no_disease_predictions,

        best_model=best_model_name,

        metrics=metrics_data
    )


# ---------------------------------------
# History
# ---------------------------------------

@app.route("/history")
def history():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ---------------------------------------
# Delete History
# ---------------------------------------

@app.route(
    "/history/delete/<int:prediction_id>"
)
def delete_history(prediction_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM predictions
        WHERE id = ?
        """,
        (prediction_id,)
    )

    connection.commit()

    connection.close()

    return redirect(
        url_for("history")
    )


# ---------------------------------------
# CSV Export
# ---------------------------------------

@app.route("/export/csv")
def export_csv():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    connection.close()

    output = io.StringIO()

    writer = csv.writer(
        output
    )

    if records:

        writer.writerow(
            records[0].keys()
        )

        for row in records:

            writer.writerow(
                list(row)
            )

    else:

        writer.writerow(
            ["No prediction history"]
        )

    output.seek(0)

    return send_file(
        io.BytesIO(
            output.getvalue().encode(
                "utf-8"
            )
        ),

        mimetype="text/csv",

        as_attachment=True,

        download_name="prediction_history.csv"
    )


# ---------------------------------------
# PDF Report
# ---------------------------------------

@app.route(
    "/report/<int:prediction_id>"
)
def generate_report(prediction_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        WHERE id = ?
        """,
        (prediction_id,)
    )

    record = cursor.fetchone()

    connection.close()

    if record is None:

        return "Prediction not found", 404

    file_path = os.path.join(
        REPORT_DIRECTORY,
        f"heart_prediction_{prediction_id}.pdf"
    )

    pdf = canvas.Canvas(
        file_path,
        pagesize=A4
    )

    width, height = A4

    y = height - 60

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        y,
        "Heart Disease Prediction Report"
    )

    y -= 45

    pdf.setFont(
        "Helvetica",
        11
    )

    pdf.drawString(
        50,
        y,
        f"Prediction ID: {record['id']}"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"Date: {record['created_at']}"
    )

    y -= 20

    pdf.drawString(
        50,
        y,
        f"Model: {record['model_name']}"
    )

    y -= 40

    pdf.setFont(
        "Helvetica-Bold",
        15
    )

    pdf.drawString(
        50,
        y,
        f"Result: {record['result']}"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        y,
        f"Estimated model probability: "
        f"{record['probability']}%"
    )

    y -= 40

    pdf.setFont(
        "Helvetica-Bold",
        13
    )

    pdf.drawString(
        50,
        y,
        "Input Values"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        10
    )

    fields = [
        ("Age", record["age"]),
        ("Sex", record["sex"]),
        ("Chest Pain", record["cp"]),
        ("Resting BP", record["trestbps"]),
        ("Cholesterol", record["chol"]),
        ("Fasting Blood Sugar", record["fbs"]),
        ("Resting ECG", record["restecg"]),
        ("Maximum Heart Rate", record["thalach"]),
        ("Exercise Angina", record["exang"]),
        ("Oldpeak", record["oldpeak"]),
        ("Slope", record["slope"]),
        ("CA", record["ca"]),
        ("Thalassemia", record["thal"])
    ]

    for label, value in fields:

        pdf.drawString(
            60,
            y,
            f"{label}: {value}"
        )

        y -= 18

        if y < 70:

            pdf.showPage()

            y = height - 60

            pdf.setFont(
                "Helvetica",
                10
            )

    y -= 20

    pdf.setFont(
        "Helvetica-Oblique",
        9
    )

    pdf.drawString(
        50,
        y,
        "Academic ML project only. "
        "This output is not a medical diagnosis."
    )

    pdf.save()

    return send_file(
        file_path,
        as_attachment=True
    )


# ---------------------------------------
# Model information
# ---------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html",

        best_model=best_model_name,

        metrics=metrics_data
    )


# ---------------------------------------
# API
# ---------------------------------------

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    try:

        data = request.get_json()

        input_data = create_input_data(
            data
        )

        prediction, result, probability = make_prediction(
            input_data
        )

        values = input_data.iloc[0].to_dict()

        prediction_id = save_prediction(
            values,
            prediction,
            result,
            probability
        )

        return jsonify({

            "success": True,

            "prediction_id": prediction_id,

            "prediction": prediction,

            "result": result,

            "probability": probability,

            "model": best_model_name

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# ---------------------------------------
# Run
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )