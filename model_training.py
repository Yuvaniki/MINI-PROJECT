# ===============================
# model_training.py — Multi-Disease Model Trainer (Final Fixed Version)
# ===============================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# ===============================
# 1️⃣ Load Dataset
# ===============================
try:
    # Load dataset with explicit data types to avoid conversion issues
    data = pd.read_csv('chronic_disease_combined.csv', dtype={
        'age': 'float64',
        'bmi': 'float64',
        'hypertension': 'int64',
        'heart_disease': 'int64',
        'diabetes': 'int64'
    })
    
    # Take a smaller subset for testing
    data = data.sample(n=min(1000, len(data)), random_state=42)
    print("✅ Dataset loaded successfully!")
    print(f"Sample size: {len(data)} records")
except FileNotFoundError:
    print("❌ Error: 'chronic_disease_combined.csv' not found. Place it in the same folder as this script.")
    exit()
except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    exit()

# ===============================
# 2️⃣ Normalize Column Names
# ===============================
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

# ===============================
# 3️⃣ Clean Data (Fix Mixed Types)
# ===============================
# Convert all categorical-like columns to strings
for col in data.columns:
    if data[col].dtype == 'object' or data[col].apply(lambda x: isinstance(x, str)).any():
        data[col] = data[col].astype(str).str.strip()

# Ensure numeric columns are properly numeric
for col in ["age", "bmi"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Ensure disease columns exist and are integers (0/1)
for col in ["hypertension", "diabetes", "heart_disease"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
    else:
        print(f"⚠️ Column '{col}' not found in dataset!")

print(f"📊 Dataset shape after cleaning: {data.shape}")

# ===============================
# Remove unused / biomarker columns and align features with the app
# ===============================
# Columns produced in some dataset versions but NOT collected by the web form
drop_cols = [
    'biomarker_a', 'biomarker_b', 'biomarker_c', 'biomarker_d',
    'smoking_history', 'hba1c_level', 'blood_glucose_level', 'target'
]
for c in drop_cols:
    if c in data.columns:
        data.drop(columns=[c], inplace=True)

# Define the feature set we expect the app to collect / send
feature_columns = [
    'age', 'gender', 'bmi', 'blood_pressure',
    'cholesterol_level', 'glucose_level', 'physical_activity',
    'smoking_status', 'alcohol_intake', 'family_history'
]

# Keep only available features from the dataset (warn if any are missing)
available_features = [c for c in feature_columns if c in data.columns]
missing_features = [c for c in feature_columns if c not in available_features]
if missing_features:
    print(f"⚠️ Warning: Missing features in dataset: {missing_features}. Using available: {available_features}")
else:
    print("✅ All required features present in dataset.")

# ===============================
# 4️⃣ Define Disease Targets
# ===============================
disease_targets = {
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "heart": "heart_disease"
}

# ===============================
# 5️⃣ Define Models
# ===============================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),  # Reduced number of trees
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)  # Limited tree depth
}

# ===============================
# 6️⃣ Train Each Disease Model
# ===============================
os.makedirs("models", exist_ok=True)
comparison_summary = []

for disease_key, disease_col in disease_targets.items():
    if disease_col not in data.columns:
        print(f"⚠️ Skipping {disease_col}: column not found in dataset.")
        continue

    print(f"\n==============================")
    print(f"🩺 Training models for {disease_col}")
    print(f"==============================")

    # Split features/labels — use only available_features determined earlier
    # (this ensures the pipeline expects the same columns the app will send)
    X = data[available_features].copy()
    y = data[disease_col]

    # Identify categorical/numeric features
    categorical_features = [col for col in X.columns if X[col].dtype == 'object']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Preprocessing with imputers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model = None
    best_accuracy = 0
    best_model_name = ""
    disease_results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        disease_results[name] = [acc, prec, rec, f1]

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_model_name = name

    # Save best model
    model_path = f"models/{disease_key}_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"💾 Saved best {disease_col} model → {model_path}")
    print(f"🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    # Append to summary
    for name, vals in disease_results.items():
        comparison_summary.append({
            "Disease": disease_col,
            "Model": name,
            "Accuracy": vals[0],
            "Precision": vals[1],
            "Recall": vals[2],
            "F1_Score": vals[3]
        })

# ===============================
# 7️⃣ Save Comparison Results
# ===============================
summary_df = pd.DataFrame(comparison_summary)
summary_df.to_csv("model_comparison_results.csv", index=False)
print("\n📁 Model comparison results saved as 'model_comparison_results.csv'")
print("\n✅ Training complete for all diseases!")
