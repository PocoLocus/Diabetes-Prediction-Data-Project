from flask import Flask, request, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import InputRequired, NumberRange
import numpy as np
import pandas as pd
import joblib
import sklearn

app = Flask(__name__)
app.config["SECRET_KEY"] = "dgrgvcebwcsdnav"

class DiabetesForm(FlaskForm):
    gender = SelectField("Gender", choices=[("female", "Female"), ("male", "Male"), ("other", "Other")], validators=[InputRequired()])
    age = IntegerField("Age", validators=[InputRequired(), NumberRange(min=1, max=100)])
    hypertension = SelectField("Hypertension", choices=[(0, "No"), (1, "Yes")], validators=[InputRequired()])
    heart_disease = SelectField("Heart disease", choices=[(0, "No"), (1, "Yes")], validators=[InputRequired()])
    smoking_history = SelectField("Smoking history", choices=[("never", "Never smoked"), ("past", "Past smoker"), ("current", "Current smoker")],
                                  validators=[InputRequired()])
    bmi = FloatField("Body Mass Index (BMI)", validators=[InputRequired(), NumberRange(min=1)])
    HbA1c_level = FloatField("Hemoglobin A1c level", validators=[InputRequired(), NumberRange(min=0)])
    blood_glucose_level = FloatField("Blood glucose level", validators=[InputRequired(), NumberRange(min=0)])
    submit = SubmitField('Submit')

# Load the Machine Learning model
loaded_model = joblib.load("../random_forest_model.pkl")

def transform_and_calculate(patient):
    # Encoding gender
    if patient["gender"] == "female":
        patient["gender_Male"] = 0
        patient["gender_Other"] = 0
    elif patient["gender"] == "male":
        patient["gender_Male"] = 1
        patient["gender_Other"] = 0
    else:
        patient["gender_Male"] = 0
        patient["gender_Other"] = 1
    # Encoding smoking history
    if patient["smoking_history"] == "never":
        patient["smoking_history_never"] = 1
        patient["smoking_history_past"] = 0
        patient["smoking_history_unknown"] = 0
    elif patient["smoking_history"] == "past":
        patient["smoking_history_never"] = 0
        patient["smoking_history_past"] = 1
        patient["smoking_history_unknown"] = 0
    else:
        patient["smoking_history_never"] = 0
        patient["smoking_history_past"] = 0
        patient["smoking_history_unknown"] = 0
    X_patient = np.array([[patient["age"], patient["hypertension"], patient["heart_disease"], patient["bmi"], patient["HbA1c_level"], patient["blood_glucose_level"],
                           patient["gender_Male"], patient["gender_Other"],
                           patient["smoking_history_never"], patient["smoking_history_past"], patient["smoking_history_unknown"]]])
    # Scale data
    loaded_scaler = joblib.load("../scaler.pkl")
    num_features_idx = [0, 3, 4, 5]
    X_patient[:, num_features_idx] = loaded_scaler.transform(X_patient[:, num_features_idx])
    # Calculate the result
    prediction = loaded_model.predict(X_patient)
    proba = loaded_model.predict_proba(X_patient)
    return proba[0][1]

@app.route("/", methods=["GET", "POST"])
def home():
    form = DiabetesForm()
    if form.validate_on_submit():
        patient = {"gender": form.gender.data,
                   "age": form.age.data,
                   "hypertension": int(form.hypertension.data),
                   "heart_disease": int(form.heart_disease.data),
                   "smoking_history": form.smoking_history.data,
                   "bmi": form.bmi.data,
                   "HbA1c_level": form.HbA1c_level.data,
                   "blood_glucose_level": form.blood_glucose_level.data}
        proba = transform_and_calculate(patient)
        return redirect(url_for("result", proba=proba))
    return render_template("home.html", form=form)

@app.route("/result")
def result():
    proba = np.round(float(request.args["proba"])*100, decimals=1)
    return render_template("result.html", proba=proba)


if __name__ == "__main__":
    app.run(debug=True)