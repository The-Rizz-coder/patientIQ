import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
import os

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/diabetic_data.csv'
OUTPUT_PATH = 'outputs/plots/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Plot Style ─────────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='muted')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PATIENTIQ — EDA PIPELINE")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
df.replace('?', np.nan, inplace=True)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)

print("\n── Missing Values ──")
print(missing_df)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x=missing_df.index, y='Missing %', data=missing_df, palette='Reds_r', ax=ax)
ax.set_title('Missing Value Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature')
ax.set_ylabel('Missing %')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}missing_values.png')
plt.close()
print("✅ Saved: missing_values.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════════════
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
readmit_rate = df['readmitted_binary'].mean() * 100
print(f"\n── Target Variable ──")
print(df['readmitted'].value_counts())
print(f"\n30-Day Readmission Rate: {readmit_rate:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['readmitted'].value_counts().plot(
    kind='bar', ax=axes[0],
    color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black'
)
axes[0].set_title('Readmission Categories', fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)

df['readmitted_binary'].value_counts().plot(
    kind='pie', ax=axes[1],
    labels=['Not Readmitted', 'Readmitted <30 days'],
    autopct='%1.1f%%',
    colors=['#2ecc71', '#e74c3c'],
    startangle=90
)
axes[1].set_title('30-Day Readmission (Binary)', fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}target_distribution.png')
plt.close()
print("✅ Saved: target_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. DEMOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
             '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df.groupby('age')['readmitted_binary'].mean().reindex(age_order).mul(100).plot(
    kind='bar', ax=axes[0], color='#3498db', edgecolor='black'
)
axes[0].set_title('Readmission Rate by Age Group', fontweight='bold')
axes[0].set_ylabel('Readmission Rate (%)')
axes[0].tick_params(axis='x', rotation=45)

df.groupby('gender')['readmitted_binary'].mean().mul(100).plot(
    kind='bar', ax=axes[1], color='#e74c3c', edgecolor='black'
)
axes[1].set_title('Readmission Rate by Gender', fontweight='bold')
axes[1].set_ylabel('Readmission Rate (%)')
axes[1].tick_params(axis='x', rotation=0)

df.groupby('race')['readmitted_binary'].mean().mul(100).sort_values(ascending=False).plot(
    kind='bar', ax=axes[2], color='#9b59b6', edgecolor='black'
)
axes[2].set_title('Readmission Rate by Race', fontweight='bold')
axes[2].set_ylabel('Readmission Rate (%)')
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle('Demographic Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}demographic_analysis.png')
plt.close()
print("✅ Saved: demographic_analysis.png")

# ══════════════════════════════════════════════════════════════════════════
# 5. CLINICAL FEATURES
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

clinical_features = [
    ('time_in_hospital',   'Time in Hospital (days)'),
    ('num_lab_procedures', 'Number of Lab Procedures'),
    ('num_medications',    'Number of Medications'),
    ('number_diagnoses',   'Number of Diagnoses'),
]

for ax, (col, label) in zip(axes.flatten(), clinical_features):
    for val, name, color in [(0, 'Not Readmitted', '#2ecc71'), (1, 'Readmitted', '#e74c3c')]:
        df[df['readmitted_binary'] == val][col].plot(kind='kde', ax=ax, label=name, color=color)
    ax.set_title(f'{label} by Readmission Status', fontweight='bold')
    ax.set_xlabel(label)
    ax.legend()

plt.suptitle('Clinical Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}clinical_features.png')
plt.close()
print("✅ Saved: clinical_features.png")

# ══════════════════════════════════════════════════════════════════════════
# 6. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════
num_cols = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses', 'readmitted_binary'
]

fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
sns.heatmap(
    df[num_cols].corr(), mask=mask, annot=True, fmt='.2f',
    cmap='coolwarm', center=0, linewidths=0.5, ax=ax
)
ax.set_title('Correlation Heatmap — Clinical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}correlation_heatmap.png')
plt.close()
print("✅ Saved: correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# 7. CHI-SQUARE TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n── Chi-Square Tests ──")
cat_features = ['age', 'gender', 'race', 'admission_type_id',
                'discharge_disposition_id', 'diabetesMed', 'change']

chi2_results = []
for col in cat_features:
    contingency = pd.crosstab(df[col], df['readmitted_binary'])
    chi2, p, dof, _ = chi2_contingency(contingency)
    chi2_results.append({
        'Feature': col,
        'Chi2': round(chi2, 2),
        'P-Value': round(p, 5),
        'Significant': '✅ Yes' if p < 0.05 else '❌ No'
    })

print(pd.DataFrame(chi2_results).sort_values('Chi2', ascending=False).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 8. KPI SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n── Hospital KPIs ──")
kpis = {
    'Avg Time in Hospital (days)':  df['time_in_hospital'].mean(),
    'Avg Lab Procedures':           df['num_lab_procedures'].mean(),
    'Avg Medications':              df['num_medications'].mean(),
    'Avg Prior Inpatient Visits':   df['number_inpatient'].mean(),
    '30-Day Readmission Rate (%)':  df['readmitted_binary'].mean() * 100,
}
for k, v in kpis.items():
    print(f"  {k}: {v:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 9. SAVE CLEANED DATASET
# ══════════════════════════════════════════════════════════════════════════
df.to_csv('data/diabetic_data_cleaned.csv', index=False)
print("\n✅ Cleaned dataset saved: data/diabetic_data_cleaned.csv")
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)