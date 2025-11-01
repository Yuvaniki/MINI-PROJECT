# ================================
# app.py — Final Version
# ================================
from flask import Flask, render_template, request, send_file, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import joblib
import numpy as np
import pandas as pd
import json
from fpdf import FPDF
from datetime import datetime
import os

# ================================
# Initialize Flask app
# ================================
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # for session handling

# ================================
# Configure MySQL connection (XAMPP)
# ================================
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # default XAMPP has no password
app.config['MYSQL_DB'] = 'chronic_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL
mysql = MySQL(app)


# ================================
# Home route
# ================================
@app.route('/')
def home():
    return render_template('home.html')

# ================================
# Login route
# ================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            return redirect(url_for('predict'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

# ================================
# Register route
# ================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        else:
            cursor.execute('INSERT INTO users (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    return render_template('register.html', msg=msg)

# ================================
# Forgot Password route
# ================================
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    msg = ''
    if request.method == 'POST':
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()
        if account:
            # In a real application, you would:
            # 1. Generate a password reset token
            # 2. Send it to the user's email
            # 3. Create a reset password page
            msg = 'Password reset instructions have been sent to your email.'
        else:
            msg = 'Email address not found!'
    return render_template('forgot_password.html', msg=msg)

# ================================
# Predict route
# ================================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # ---- Collect and preprocess form data ----
            try:
                # Convert all inputs to appropriate numeric values
                age = float(request.form['age'])
                
                # Gender: Male = 1, Female = 0
                gender_map = {'Male': 1, 'Female': 0}
                gender = gender_map[request.form['gender']]
                
                bmi = float(request.form['bmi'])
                bp = float(request.form['bp'])
                cholesterol = float(request.form['cholesterol'])
                glucose = float(request.form['glucose'])
                physical_activity = float(request.form['physical_activity'])
                
                # Smoking: Never = 0, Former = 1, Current = 2
                smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
                smoking = smoking_map[request.form['smoking']]
                
                alcohol = float(request.form['alcohol'])
                
                # Family History: No = 0, Yes = 1
                family_history_map = {'No': 0, 'Yes': 1}
                family_history = family_history_map[request.form['family_history']]

            except ValueError as e:
                return render_template('predict.html', 
                                    error="Please ensure all numeric fields contain valid numbers.")
            except KeyError as e:
                return render_template('predict.html', 
                                    error="Please ensure all fields are filled out correctly.")

            # ---- Prepare features as a DataFrame with exact training column names ----
            features = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'bmi': [bmi],
                'blood_pressure': [bp],
                'cholesterol_level': [cholesterol],
                'glucose_level': [glucose],
                'physical_activity': [physical_activity],
                'smoking_status': [smoking],
                'alcohol_intake': [alcohol],
                'family_history': [family_history]
            })

            # ---- Load ML models ----
            models = {
                "Hypertension": joblib.load("models/hypertension_model.joblib"),
                "Diabetes": joblib.load("models/diabetes_model.joblib"),
                "Heart Disease": joblib.load("models/heart_model.joblib")
            }

            # ---- Ensure features align with each model's expected input ----
            predictions = {}
            for disease, model in models.items():
                # Determine expected input columns from the model's preprocessor if available
                expected_cols = None
                try:
                    pre = model.named_steps.get('preprocessor')
                    if pre is not None and hasattr(pre, 'feature_names_in_'):
                        expected_cols = list(pre.feature_names_in_)
                except Exception:
                    expected_cols = None

                if expected_cols is None:
                    # fallback to pipeline-level attribute
                    if hasattr(model, 'feature_names_in_'):
                        expected_cols = list(model.feature_names_in_)

                if expected_cols is None:
                    # as last resort, use the current DataFrame columns
                    expected_cols = list(features.columns)

                # Add any missing expected columns with sensible default values (0)
                missing = [c for c in expected_cols if c not in features.columns]
                if missing:
                    for c in missing:
                        # default numeric 0; if it's gender or similar, 0 is a safe default
                        features[c] = 0

                # Reorder columns to match model expectation
                features_for_model = features.reindex(columns=expected_cols)

                # Ensure numeric types where possible
                for col in features_for_model.columns:
                    try:
                        features_for_model[col] = features_for_model[col].astype(float)
                    except Exception:
                        # keep original dtype for categorical columns
                        pass

                # ---- Make prediction for this model ----
                prob = model.predict_proba(features_for_model)[0][1] * 100
                predictions[disease] = round(prob, 2)

            # ---- Create result message ----
            result_message = "<br>".join([f"<b>{d}</b>: {p}%" for d, p in predictions.items()])

            # ---- Save results temporarily ----
            with open("temp_report.json", "w") as f:
                json.dump(predictions, f)

            return render_template('predict.html',
                                   result=result_message,
                                   show_download=True)

        except Exception as e:
            return render_template('predict.html',
                                   result=f"Error during prediction: {e}",
                                   show_download=False)

    # GET request → just show form
    return render_template('predict.html', result=None, show_download=False)


# ================================
# Download PDF Report
# ================================
@app.route('/download_report')
def download_report():
    try:
        with open("temp_report.json", "r") as f:
            predictions = json.load(f)

        os.makedirs("reports", exist_ok=True)
        file_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = os.path.join("reports", file_name)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Chronic Disease Prediction Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)

        for disease, prob in predictions.items():
            pdf.cell(200, 10, f"{disease}: {prob}%", ln=True)

        pdf.output(file_path)

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return f"Error generating report: {e}"


# ================================
# Run Flask app
# ================================
# Create tables if they don't exist
def init_db():
    cursor = mysql.connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(100) NOT NULL
        )
    ''')
    mysql.connection.commit()
    cursor.close()

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True)
