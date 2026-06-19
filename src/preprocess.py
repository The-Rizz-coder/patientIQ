import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/diabetic_data_cleaned.csv'
OUTPUT_PATH = 'data/diabetic_data_processed.csv'
MODEL_PATH  = 'models/preprocessor.pkl'
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("PATIENTIQ — PREPROCESSING + FEATURE ENGINEERING")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════
# 2. DROP IRRELEVANT / HIGH MISSING COLUMNS
# ══════════════════════════════════════════════════════════════════════════
drop_cols = [
    'encounter_id',       # ID — not a feature
    'patient_nbr',        # ID — not a feature
    'weight',             # 96.86% missing
    'max_glu_serum',      # 94.75% missing
    'A1Cresult',          # 83.28% missing
    'medical_specialty',  # 49.08% missing
    'payer_code',         # 39.56% missing — billing, not clinical
    'examide',            # near-zero variance
    'citoglipton',        # near-zero variance
    'glimepiride-pioglitazone',    # near-zero variance
    'metformin-rosiglitazone',     # near-zero variance
    'metformin-pioglitazone',      # near-zero variance
]

df.drop(columns=drop_cols, inplace=True)
print(f"✅ Dropped {len(drop_cols)} irrelevant/high-missing columns")
print(f"   Remaining: {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════
# 3. HANDLE REMAINING MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════
df['race'].fillna('Unknown', inplace=True)
df['diag_1'].fillna('Unknown', inplace=True)
df['diag_2'].fillna('Unknown', inplace=True)
df['diag_3'].fillna('Unknown', inplace=True)
print("✅ Imputed remaining missing values")

# ══════════════════════════════════════════════════════════════════════════
# 4. HANDLE INVALID GENDER VALUES
# ══════════════════════════════════════════════════════════════════════════
df = df[df['gender'] != 'Unknown/Invalid']
print(f"✅ Removed invalid gender rows → {df.shape[0]:,} rows remaining")

# ══════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════
print("\n── Feature Engineering ──")

# 5.1 Age → Numeric Midpoint
age_map = {
    '[0-10)': 5,   '[10-20)': 15,  '[20-30)': 25,
    '[30-40)': 35, '[40-50)': 45,  '[50-60)': 55,
    '[60-70)': 65, '[70-80)': 75,  '[80-90)': 85,
    '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_map)
print("✅ age → age_numeric (midpoint)")

# 5.2 Age Risk Tier
def age_risk_tier(age):
    if age < 40:
        return 'Young'
    elif age < 60:
        return 'Middle-Aged'
    elif age < 75:
        return 'Senior'
    else:
        return 'Elderly'

df['age_tier'] = df['age_numeric'].apply(age_risk_tier)
print("✅ age_tier engineered (Young / Middle-Aged / Senior / Elderly)")

# 5.3 Total Prior Visits
df['total_prior_visits'] = (
    df['number_outpatient'] +
    df['number_emergency'] +
    df['number_inpatient']
)
print("✅ total_prior_visits engineered")

# 5.4 High Utilizer Flag
df['high_utilizer'] = (df['total_prior_visits'] > 3).astype(int)
print("✅ high_utilizer flag engineered")

# 5.5 Medication Complexity Score
med_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin'
]
# Count medications that are not 'No'
df['medication_complexity'] = df[med_cols].apply(
    lambda row: sum(1 for v in row if v != 'No'), axis=1
)
print("✅ medication_complexity score engineered")

# 5.6 Medication Change Flag
df['med_changed'] = (df['change'] == 'Ch').astype(int)
print("✅ med_changed flag engineered")

# 5.7 Diagnosis Grouping (Primary Diagnosis)
def group_diagnosis(diag):
    try:
        code = str(diag)
        if code == 'Unknown':
            return 'Unknown'
        if code.startswith('V') or code.startswith('E'):
            return 'External'
        val = float(code)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif 250 <= val <= 250.99:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'

df['diag_1_group'] = df['diag_1'].apply(group_diagnosis)
df['diag_2_group'] = df['diag_2'].apply(group_diagnosis)
df['diag_3_group'] = df['diag_3'].apply(group_diagnosis)
print("✅ diag_1/2/3 grouped into clinical categories")

# 5.8 Readmission Risk Score (domain heuristic)
df['risk_score'] = (
    df['number_inpatient'] * 3 +
    df['number_emergency'] * 2 +
    df['number_outpatient'] * 1 +
    df['time_in_hospital'] * 0.5 +
    df['medication_complexity'] * 0.3
)
print("✅ risk_score heuristic engineered")

# ══════════════════════════════════════════════════════════════════════════
# 6. TARGET ENCODING
# ══════════════════════════════════════════════════════════════════════════
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
df.drop(columns=['readmitted'], inplace=True)
print("\n✅ Target encoded: readmitted_binary (1 = readmitted <30 days)")

# ══════════════════════════════════════════════════════════════════════════
# 7. DEFINE FEATURE SETS
# ══════════════════════════════════════════════════════════════════════════
drop_from_model = ['age', 'diag_1', 'diag_2', 'diag_3']
df.drop(columns=drop_from_model, inplace=True)

numerical_features = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'total_prior_visits', 'medication_complexity', 'risk_score'
]

categorical_features = [
    'race', 'gender', 'age_tier', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id',
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'change', 'diabetesMed'
]

binary_features = ['high_utilizer', 'med_changed']

print(f"\n── Feature Summary ──")
print(f"  Numerical features:   {len(numerical_features)}")
print(f"  Categorical features: {len(categorical_features)}")
print(f"  Binary features:      {len(binary_features)}")

# ══════════════════════════════════════════════════════════════════════════
# 8. BUILD SKLEARN PIPELINE
# ══════════════════════════════════════════════════════════════════════════
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline,  numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('bin', 'passthrough',        binary_features)
])

print("\n✅ Scikit-learn Pipeline + ColumnTransformer built")

# ══════════════════════════════════════════════════════════════════════════
# 9. FIT + TRANSFORM
# ══════════════════════════════════════════════════════════════════════════
X = df[numerical_features + categorical_features + binary_features]
y = df['readmitted_binary']

X_transformed = preprocessor.fit_transform(X)
print(f"✅ Preprocessing applied → shape: {X_transformed.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════
joblib.dump(preprocessor, MODEL_PATH)
print(f"✅ Preprocessor saved: {MODEL_PATH}")

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Processed dataset saved: {OUTPUT_PATH}")

print(f"\n── Final Dataset ──")
print(f"  Shape:               {df.shape}")
print(f"  Readmission Rate:    {y.mean()*100:.2f}%")
print(f"  Class distribution:  {y.value_counts().to_dict()}")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)