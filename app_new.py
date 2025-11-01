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

            # ---- Prepare features as a DataFrame with the same structure as training data ----
            features_dict = {
                'age': [age],
                'gender': [gender],
                'bmi': [bmi],
                'blood_pressure': [bp],
                'cholesterol_level': [cholesterol],
                'glucose_level': [glucose],
                'physical_activity': [physical_activity],
                'smoking_status': [smoking],
                'alcohol_intake': [alcohol],
                'family_history': [family_history],
                'biomarker_a': [55.0],  # Using average values for biomarkers
                'biomarker_b': [100.0],
                'biomarker_c': [75.0],
                'biomarker_d': [120.0],
                'hba1c_level': [5.5],  # Default value
                'blood_glucose_level': [glucose],  # Using glucose level provided
                'target': [0]  # Placeholder value
            }
            
            features = pd.DataFrame(features_dict)

            # ---- Load ML models ----
            models = {
                "Hypertension": joblib.load("models/hypertension_model.joblib"),
                "Diabetes": joblib.load("models/diabetes_model.joblib"),
                "Heart Disease": joblib.load("models/heart_model.joblib")
            }

            # ---- Make predictions ----
            predictions = {}
            for disease, model in models.items():
                prob = model.predict_proba(features)[0][1] * 100
                predictions[disease] = round(prob, 2)

            # ---- Create result message ----
            result_message = "<br>".join([f"<b>{d}</b>: {p}% risk" for d, p in predictions.items()])

            # ---- Save results temporarily ----
            with open("temp_report.json", "w") as f:
                json.dump(predictions, f)

            return render_template('predict.html',
                               result=result_message,
                               show_download=True)

        except Exception as e:
            return render_template('predict.html',
                               error=f"Error during prediction: {str(e)}")

    return render_template('predict.html')

# ================================
# Download Report route
# ================================
@app.route('/download_report')
def download_report():
    try:
        with open('temp_report.json', 'r') as f:
            predictions = json.load(f)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Chronic Disease Risk Assessment Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Date
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        pdf.ln(10)
        
        # Results
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Prediction Results:', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        for disease, probability in predictions.items():
            pdf.cell(0, 10, f'{disease}: {probability}% risk', 0, 1)
        
        # Save the PDF
        pdf.output('disease_report.pdf')
        
        return send_file('disease_report.pdf', as_attachment=True)
    except Exception as e:
        return str(e)

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