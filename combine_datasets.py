# ============================================
# combine_datasets.py
# ============================================

import pandas as pd

# ===============================
# Load both datasets
# ===============================
chronic = pd.read_csv("chronic_disease_dataset.csv")
diabetes = pd.read_csv("diabetes_prediction_dataset.csv")

print("✅ Datasets loaded successfully!")
print(f"Chronic Disease Dataset: {chronic.shape}")
print(f"Diabetes Dataset: {diabetes.shape}")

# ===============================
# Normalize column names
# ===============================
chronic.columns = chronic.columns.str.strip().str.lower()
diabetes.columns = diabetes.columns.str.strip().str.lower()

# ===============================
# Ensure consistent disease columns
# (only hypertension, diabetes, heart_disease)
# ===============================
for col in ["hypertension", "diabetes", "heart_disease"]:
    if col not in chronic.columns:
        chronic[col] = 0
    if col not in diabetes.columns:
        diabetes[col] = 0

# ===============================
# Identify common columns (for info)
# ===============================
common_cols = list(set(chronic.columns) & set(diabetes.columns))
print(f"🔗 Common columns: {common_cols}")

# ===============================
# Combine datasets (stack rows)
# ===============================
combined = pd.concat([chronic, diabetes], ignore_index=True)
combined.drop_duplicates(inplace=True)

# ===============================
# Save combined dataset
# ===============================
combined.to_csv("chronic_disease_combined.csv", index=False)
print("💾 Combined dataset saved as 'chronic_disease_combined.csv'")
print(f"✅ Final shape: {combined.shape}")
