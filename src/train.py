import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

DATA_PATH   = 'data/diabetic_data_processed.csv'
OUTPUT_PATH = 'outputs/plots/'
MODEL_PATH  = 'models/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH,  exist_ok=True)

print("=" * 60)
print("PATIENTIQ — MODEL TRAINING v3 (FEATURE SELECTED)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
preprocessor = joblib.load('models/preprocessor.pkl')

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
feature_cols    = numerical_features + categorical_features + binary_features

X_raw = df[feature_cols]
y     = df['readmitted_binary']
X     = preprocessor.transform(X_raw)

print(f"✅ Data loaded → X: {X.shape} | Readmission rate: {y.mean()*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# 2. TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE applied → Train: {X_train_sm.shape[0]:,} balanced samples")

# ══════════════════════════════════════════════════════════════════════════
# 3. FIND OPTIMAL THRESHOLD
# ══════════════════════════════════════════════════════════════════════════
def find_best_threshold(model, X_te, y_te):
    probs = model.predict_proba(X_te)[:, 1]
    thresholds = np.arange(0.1, 0.6, 0.02)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_te, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t
    return best_thresh, best_f1

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, threshold=None):
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]

    if threshold is None:
        threshold, _ = find_best_threshold(model, X_te, y_te)

    y_pred    = (y_prob >= threshold).astype(int)
    acc       = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall    = recall_score(y_te, y_pred, zero_division=0)
    f1        = f1_score(y_te, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_te, y_prob)

    print(f"\n── {name} (threshold={threshold:.2f}) ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(classification_report(y_te, y_pred,
          target_names=['Not Readmitted','Readmitted'], zero_division=0))

    return model, y_pred, y_prob, threshold, {
        'name': name, 'threshold': round(threshold, 2),
        'accuracy': acc, 'precision': precision,
        'recall': recall, 'f1': f1, 'roc_auc': roc_auc
    }

# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model, lr_pred, lr_prob, lr_thresh, lr_metrics = evaluate_model(
    'Logistic Regression', lr, X_train_sm, y_train_sm, X_test, y_test
)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_split=20,
    min_samples_leaf=10, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf_model, rf_pred, rf_prob, rf_thresh, rf_metrics = evaluate_model(
    'Random Forest', rf, X_train_sm, y_train_sm, X_test, y_test
)

xgb_inst = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model, xgb_pred, xgb_prob, xgb_thresh, xgb_metrics = evaluate_model(
    'XGBoost', xgb_inst, X_train_sm, y_train_sm, X_test, y_test
)

# ══════════════════════════════════════════════════════════════════════════
# 5. CV ON BEST MODEL (SMOTE INSIDE PIPELINE)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Stratified K-Fold CV (XGBoost, SMOTE inside pipeline) ──")
cv_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', random_state=42, n_jobs=-1
    ))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    cv_pipeline, X_train, y_train,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"  CV ROC-AUC: {cv_scores.round(4)}")
print(f"  Mean: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n── Model Comparison ──")
comparison = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
print(comparison.set_index('name').round(4).to_string())

# ══════════════════════════════════════════════════════════════════════════
# 7. ROC CURVE
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
for name, prob, color in [
    ('Logistic Regression', lr_prob,  '#3498db'),
    ('Random Forest',       rf_prob,  '#2ecc71'),
    ('XGBoost',             xgb_prob, '#e74c3c'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=color, lw=2)

ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}roc_curve.png')
plt.close()
print("\n✅ Saved: roc_curve.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, pred) in zip(axes, [
    ('Logistic Regression', lr_pred),
    ('Random Forest',       rf_pred),
    ('XGBoost',             xgb_pred),
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Readmitted','Readmitted'],
                yticklabels=['Not Readmitted','Readmitted'])
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}confusion_matrix.png')
plt.close()
print("✅ Saved: confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════
try:
    cat_names = preprocessor.named_transformers_['cat']['encoder']\
                .get_feature_names_out(categorical_features).tolist()
    all_names = numerical_features + cat_names + binary_features
except:
    all_names = [f'feature_{i}' for i in range(X_train_sm.shape[1])]

importances = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature':    all_names[:len(importances)],
    'Importance': importances
}).sort_values('Importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis', ax=ax)
ax.set_title('Top 20 Feature Importances — XGBoost', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}feature_importance.png')
plt.close()
print("✅ Saved: feature_importance.png")
print(feat_imp_df.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE MODELS + THRESHOLD
# ══════════════════════════════════════════════════════════════════════════
joblib.dump(lr_model,  f'{MODEL_PATH}lr_model.pkl')
joblib.dump(rf_model,  f'{MODEL_PATH}rf_model.pkl')
joblib.dump(xgb_model, f'{MODEL_PATH}xgb_model.pkl')
joblib.dump({
    'xgb_threshold': xgb_thresh,
    'rf_threshold':  rf_thresh,
    'lr_threshold':  lr_thresh
}, f'{MODEL_PATH}thresholds.pkl')

print(f"\n✅ All models + thresholds saved to {MODEL_PATH}")
print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"XGBoost → ROC-AUC: {xgb_metrics['roc_auc']:.4f} | "
      f"F1: {xgb_metrics['f1']:.4f} | "
      f"Recall: {xgb_metrics['recall']:.4f}")