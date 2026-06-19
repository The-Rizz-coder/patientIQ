from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# ── Load Artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

preprocessor   = joblib.load(os.path.join(BASE, 'models/preprocessor.pkl'))
xgb_model      = joblib.load(os.path.join(BASE, 'models/xgb_model.pkl'))
rf_model       = joblib.load(os.path.join(BASE, 'models/rf_model.pkl'))
kmeans         = joblib.load(os.path.join(BASE, 'models/kmeans.pkl'))
cluster_scaler = joblib.load(os.path.join(BASE, 'models/cluster_scaler.pkl'))
tier_map       = joblib.load(os.path.join(BASE, 'models/tier_map.pkl'))
thresholds     = joblib.load(os.path.join(BASE, 'models/thresholds.pkl'))

df = pd.read_csv(os.path.join(BASE, 'data/diabetic_data_clustered.csv'))

# ── Feature Definitions ─────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'total_prior_visits', 'medication_complexity', 'risk_score'
]
CATEGORICAL_FEATURES = [
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
BINARY_FEATURES = ['high_utilizer', 'med_changed']
FEATURE_COLS    = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

CLUSTERING_FEATURES = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'total_prior_visits', 'medication_complexity', 'risk_score'
]

# ── Dashboard Stats ──────────────────────────────────────────────────────
def get_dashboard_stats():
    total         = len(df)
    readmit_rate  = round(df['readmitted_binary'].mean() * 100, 2)
    avg_stay      = round(df['time_in_hospital'].mean(), 1)
    avg_meds      = round(df['num_medications'].mean(), 1)
    high_risk     = int((df['risk_tier'] == 'High Risk').sum())
    medium_risk   = int((df['risk_tier'] == 'Medium Risk').sum())
    low_risk      = int((df['risk_tier'] == 'Low Risk').sum())

    # Age group readmission

    age_readmit = df.groupby('age_tier')['readmitted_binary'].mean().mul(100).round(2)

    # Diagnosis group readmission
    diag_readmit = df.groupby('diag_1_group')['readmitted_binary'].mean().mul(100).sort_values(ascending=False).round(2)

    # Risk tier distribution
    tier_dist = df['risk_tier'].value_counts().reindex(['Low Risk','Medium Risk','High Risk'])

    # Readmission by time in hospital
    time_readmit = df.groupby('time_in_hospital')['readmitted_binary'].mean().mul(100).head(14).round(2)

    # Cluster profiles
    cluster_profile = df.groupby('risk_tier')[['age_numeric','time_in_hospital',
                      'num_medications','number_inpatient','risk_score',
                      'readmitted_binary']].mean().round(2)

    return {
        'total':          total,
        'readmit_rate':   readmit_rate,
        'avg_stay':       avg_stay,
        'avg_meds':       avg_meds,
        'high_risk':      high_risk,
        'medium_risk':    medium_risk,
        'low_risk':       low_risk,
        'age_labels':     age_readmit.index.tolist(),
        'age_values':     age_readmit.values.tolist(),
        'diag_labels':    diag_readmit.index.tolist(),
        'diag_values':    diag_readmit.values.tolist(),
        'tier_labels':    tier_dist.index.tolist(),
        'tier_values':    tier_dist.values.tolist(),
        'time_labels':    time_readmit.index.tolist(),
        'time_values':    time_readmit.values.tolist(),
        'cluster_profile': cluster_profile.to_dict()
    }

# ── Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats = get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # ── Build age_numeric + derived features ──
        age_numeric = int(form.get('age_numeric', 45))

        def age_risk_tier(age):
            if age < 40:   return 'Young'
            elif age < 60: return 'Middle-Aged'
            elif age < 75: return 'Senior'
            else:          return 'Elderly'

        time_in_hospital     = int(form.get('time_in_hospital', 3))
        num_lab_procedures   = int(form.get('num_lab_procedures', 40))
        num_procedures       = int(form.get('num_procedures', 1))
        num_medications      = int(form.get('num_medications', 15))
        number_outpatient    = int(form.get('number_outpatient', 0))
        number_emergency     = int(form.get('number_emergency', 0))
        number_inpatient     = int(form.get('number_inpatient', 0))
        number_diagnoses     = int(form.get('number_diagnoses', 5))
        total_prior_visits   = number_outpatient + number_emergency + number_inpatient
        high_utilizer        = int(total_prior_visits > 3)

        med_cols_list = [
            'metformin','repaglinide','nateglinide','chlorpropamide',
            'glimepiride','glipizide','glyburide','tolbutamide',
            'pioglitazone','rosiglitazone','acarbose','miglitol',
            'troglitazone','tolazamide','insulin',
            'glyburide-metformin','glipizide-metformin'
        ]
        medication_complexity = sum(
            1 for col in med_cols_list
            if form.get(col, 'No') != 'No'
        )

        change       = form.get('change', 'No')
        med_changed  = int(change == 'Ch')

        risk_score = (
            number_inpatient * 3 +
            number_emergency * 2 +
            number_outpatient * 1 +
            time_in_hospital * 0.5 +
            medication_complexity * 0.3
        )

        input_data = {
            'age_numeric':            age_numeric,
            'time_in_hospital':       time_in_hospital,
            'num_lab_procedures':     num_lab_procedures,
            'num_procedures':         num_procedures,
            'num_medications':        num_medications,
            'number_outpatient':      number_outpatient,
            'number_emergency':       number_emergency,
            'number_inpatient':       number_inpatient,
            'number_diagnoses':       number_diagnoses,
            'total_prior_visits':     total_prior_visits,
            'medication_complexity':  medication_complexity,
            'risk_score':             risk_score,
            'race':                   form.get('race', 'Caucasian'),
            'gender':                 form.get('gender', 'Female'),
            'age_tier':               age_risk_tier(age_numeric),
            'admission_type_id':      form.get('admission_type_id', '1'),
            'discharge_disposition_id': form.get('discharge_disposition_id', '1'),
            'admission_source_id':    form.get('admission_source_id', '7'),
            'diag_1_group':           form.get('diag_1_group', 'Diabetes'),
            'diag_2_group':           form.get('diag_2_group', 'Other'),
            'diag_3_group':           form.get('diag_3_group', 'Other'),
            'metformin':              form.get('metformin', 'No'),
            'repaglinide':            form.get('repaglinide', 'No'),
            'nateglinide':            form.get('nateglinide', 'No'),
            'chlorpropamide':         form.get('chlorpropamide', 'No'),
            'glimepiride':            form.get('glimepiride', 'No'),
            'glipizide':              form.get('glipizide', 'No'),
            'glyburide':              form.get('glyburide', 'No'),
            'tolbutamide':            form.get('tolbutamide', 'No'),
            'pioglitazone':           form.get('pioglitazone', 'No'),
            'rosiglitazone':          form.get('rosiglitazone', 'No'),
            'acarbose':               form.get('acarbose', 'No'),
            'miglitol':               form.get('miglitol', 'No'),
            'troglitazone':           form.get('troglitazone', 'No'),
            'tolazamide':             form.get('tolazamide', 'No'),
            'insulin':                form.get('insulin', 'No'),
            'glyburide-metformin':    form.get('glyburide-metformin', 'No'),
            'glipizide-metformin':    form.get('glipizide-metformin', 'No'),
            'change':                 change,
            'diabetesMed':            form.get('diabetesMed', 'Yes'),
            'high_utilizer':          high_utilizer,
            'med_changed':            med_changed,
        }

        input_df  = pd.DataFrame([input_data])
        X_input   = preprocessor.transform(input_df[FEATURE_COLS])
        xgb_prob  = float(xgb_model.predict_proba(X_input)[0][1])
        rf_prob   = float(rf_model.predict_proba(X_input)[0][1])
        ensemble_prob = round((xgb_prob * 0.6 + rf_prob * 0.4), 4)

        threshold = thresholds.get('xgb_threshold', 0.18)
        prediction = int(ensemble_prob >= threshold)

        # ── Risk Tier via KMeans ──
        cluster_input = np.array([[
            input_data['age_numeric'],       input_data['time_in_hospital'],
            input_data['num_lab_procedures'],input_data['num_procedures'],
            input_data['num_medications'],   input_data['number_outpatient'],
            input_data['number_emergency'],  input_data['number_inpatient'],
            input_data['number_diagnoses'],  input_data['total_prior_visits'],
            input_data['medication_complexity'], input_data['risk_score']
        ]])
        cluster_scaled = cluster_scaler.transform(cluster_input)
        cluster_id     = int(kmeans.predict(cluster_scaled)[0])
        risk_tier      = tier_map.get(cluster_id, 'Medium Risk')

        # ── Risk Label ──
        risk_pct = round(ensemble_prob * 100, 1)
        if risk_pct < 10:
            risk_label = 'Low Risk'
            risk_color = 'success'
        elif risk_pct < 20:
            risk_label = 'Medium Risk'
            risk_color = 'warning'
        else:
            risk_label = 'High Risk'
            risk_color = 'danger'

        # ── Key Risk Drivers ──
        drivers = []
        if number_inpatient >= 2:
            drivers.append({'factor': 'Prior Inpatient Visits', 'value': str(number_inpatient), 'impact': 'High'})
        if number_emergency >= 1:
            drivers.append({'factor': 'Emergency Visits', 'value': str(number_emergency), 'impact': 'High'})
        if time_in_hospital >= 7:
            drivers.append({'factor': 'Long Hospital Stay', 'value': f'{time_in_hospital} days', 'impact': 'Medium'})
        if medication_complexity >= 3:
            drivers.append({'factor': 'High Medication Complexity', 'value': str(medication_complexity), 'impact': 'Medium'})
        if number_diagnoses >= 7:
            drivers.append({'factor': 'Multiple Diagnoses', 'value': str(number_diagnoses), 'impact': 'Medium'})
        if med_changed:
            drivers.append({'factor': 'Medication Changed at Discharge', 'value': 'Yes', 'impact': 'Medium'})
        if not drivers:
            drivers.append({'factor': 'No Major Risk Factors Identified', 'value': '—', 'impact': 'Low'})

        return jsonify({
            'success':        True,
            'probability':    risk_pct,
            'prediction':     prediction,
            'risk_label':     risk_label,
            'risk_color':     risk_color,
            'risk_tier':      risk_tier,
            'xgb_prob':       round(xgb_prob * 100, 1),
            'rf_prob':        round(rf_prob * 100, 1),
            'risk_score':     round(risk_score, 2),
            'drivers':        drivers,
            'total_prior_visits': total_prior_visits,
            'med_complexity': medication_complexity,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
def dashboard():
    stats = get_dashboard_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/api/stats')
def api_stats():
    return jsonify(get_dashboard_stats())

if __name__ == '__main__':
    app.run(debug=True)