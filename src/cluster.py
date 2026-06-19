import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

DATA_PATH   = 'data/diabetic_data_processed.csv'
OUTPUT_PATH = 'outputs/plots/'
MODEL_PATH  = 'models/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("PATIENTIQ — K-MEANS CLUSTERING")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD + PREPARE CLUSTERING FEATURES
# ══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
preprocessor = joblib.load('models/preprocessor.pkl')

# Use numerical features only for clustering — interpretable clusters
clustering_features = [
    'age_numeric', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'total_prior_visits', 'medication_complexity', 'risk_score'
]

X_cluster = df[clustering_features].copy()
print(f"\n✅ Clustering features: {len(clustering_features)}")
print(f"   Dataset shape: {X_cluster.shape}")

# Scale for KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
print("✅ Features scaled")

# ══════════════════════════════════════════════════════════════════════════
# 2. ELBOW METHOD
# ══════════════════════════════════════════════════════════════════════════
print("\n── Elbow Method ──")
inertias = []
K_range  = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  K={k} | Inertia: {km.inertia_:,.0f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method — Optimal K Selection', fontsize=14, fontweight='bold')
ax.axvline(x=3, color='red', linestyle='--', label='K=3 (selected)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}elbow_curve.png')
plt.close()
print("✅ Saved: elbow_curve.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. SILHOUETTE SCORES
# ══════════════════════════════════════════════════════════════════════════
print("\n── Silhouette Scores ──")
sil_scores = []

# Use sample for speed
sample_idx = np.random.choice(len(X_scaled), size=10000, replace=False)
X_sample   = X_scaled[sample_idx]

for k in range(2, 8):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels)
    sil_scores.append(sil)
    print(f"  K={k} | Silhouette Score: {sil:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(2, 8), sil_scores, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score — Cluster Quality', fontsize=14, fontweight='bold')
ax.axvline(x=3, color='red', linestyle='--', label='K=3 (selected)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}silhouette_score.png')
plt.close()
print("✅ Saved: silhouette_score.png")

# ══════════════════════════════════════════════════════════════════════════
# 4. FINAL KMEANS — K=3
# ══════════════════════════════════════════════════════════════════════════
print("\n── Fitting Final K-Means (K=3) ──")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
df['cluster'] = kmeans.fit_predict(X_scaled)
print(f"✅ Clusters assigned")
print(f"   Distribution: {df['cluster'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════
# 5. MAP CLUSTERS TO RISK TIERS
# ══════════════════════════════════════════════════════════════════════════
# Rank clusters by readmission rate → assign Low/Medium/High
cluster_readmit = df.groupby('cluster')['readmitted_binary'].mean()
rank            = cluster_readmit.rank().astype(int)
tier_map        = {cluster: ['Low Risk', 'Medium Risk', 'High Risk'][r-1]
                   for cluster, r in rank.items()}

df['risk_tier'] = df['cluster'].map(tier_map)
print(f"\n── Risk Tier Mapping ──")
for cluster, tier in tier_map.items():
    rate = cluster_readmit[cluster] * 100
    size = (df['cluster'] == cluster).sum()
    print(f"  Cluster {cluster} → {tier} | Readmission Rate: {rate:.2f}% | Size: {size:,}")

# ══════════════════════════════════════════════════════════════════════════
# 6. CLUSTER PROFILES
# ══════════════════════════════════════════════════════════════════════════
print("\n── Cluster Profiles ──")
profile_cols = clustering_features + ['readmitted_binary']
profile      = df.groupby('risk_tier')[profile_cols].mean().round(2)
print(profile.T.to_string())

# ══════════════════════════════════════════════════════════════════════════
# 7. CLUSTER PROFILE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

plot_features = [
    ('age_numeric',        'Average Age'),
    ('time_in_hospital',   'Avg Time in Hospital'),
    ('num_medications',    'Avg Medications'),
    ('number_inpatient',   'Avg Prior Inpatient Visits'),
    ('medication_complexity', 'Medication Complexity'),
    ('risk_score',         'Risk Score'),
]

tier_colors = {
    'Low Risk':    '#2ecc71',
    'Medium Risk': '#f39c12',
    'High Risk':   '#e74c3c'
}

for ax, (col, label) in zip(axes.flatten(), plot_features):
    tier_means = df.groupby('risk_tier')[col].mean().reindex(
        ['Low Risk', 'Medium Risk', 'High Risk']
    )
    bars = ax.bar(
        tier_means.index, tier_means.values,
        color=[tier_colors[t] for t in tier_means.index],
        edgecolor='black'
    )
    ax.set_title(label, fontweight='bold')
    ax.set_ylabel('Mean Value')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, tier_means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Clinical Risk Tier Profiles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}cluster_profiles.png')
plt.close()
print("\n✅ Saved: cluster_profiles.png")

# ══════════════════════════════════════════════════════════════════════════
# 8. PCA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
print("── PCA Cluster Visualization ──")
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

sample_idx = np.random.choice(len(X_pca), size=5000, replace=False)
pca_df = pd.DataFrame({
    'PC1':       X_pca[sample_idx, 0],
    'PC2':       X_pca[sample_idx, 1],
    'Risk Tier': df['risk_tier'].iloc[sample_idx].values
})

fig, ax = plt.subplots(figsize=(10, 7))
for tier, color in tier_colors.items():
    mask = pca_df['Risk Tier'] == tier
    ax.scatter(
        pca_df.loc[mask, 'PC1'],
        pca_df.loc[mask, 'PC2'],
        c=color, label=tier, alpha=0.4, s=10
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Patient Risk Segments — PCA Projection', fontsize=14, fontweight='bold')
ax.legend(markerscale=3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}pca_clusters.png')
plt.close()
print("✅ Saved: pca_clusters.png")

# ══════════════════════════════════════════════════════════════════════════
# 9. READMISSION RATE BY TIER
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tier_readmit = df.groupby('risk_tier')['readmitted_binary'].mean().mul(100)\
                 .reindex(['Low Risk', 'Medium Risk', 'High Risk'])

bars = axes[0].bar(
    tier_readmit.index, tier_readmit.values,
    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black'
)
axes[0].set_title('Readmission Rate by Risk Tier', fontweight='bold')
axes[0].set_ylabel('Readmission Rate (%)')
for bar, val in zip(bars, tier_readmit.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

tier_size = df['risk_tier'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
axes[1].pie(
    tier_size.values,
    labels=tier_size.index,
    autopct='%1.1f%%',
    colors=['#2ecc71', '#f39c12', '#e74c3c'],
    startangle=90
)
axes[1].set_title('Patient Distribution by Risk Tier', fontweight='bold')

plt.suptitle('Clinical Risk Segmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}risk_segmentation.png')
plt.close()
print("✅ Saved: risk_segmentation.png")

# ══════════════════════════════════════════════════════════════════════════
# 10. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════
joblib.dump(kmeans, f'{MODEL_PATH}kmeans.pkl')
joblib.dump(scaler, f'{MODEL_PATH}cluster_scaler.pkl')
joblib.dump(tier_map, f'{MODEL_PATH}tier_map.pkl')

df.to_csv('data/diabetic_data_clustered.csv', index=False)

print(f"\n✅ K-Means model saved: {MODEL_PATH}kmeans.pkl")
print(f"✅ Cluster scaler saved: {MODEL_PATH}cluster_scaler.pkl")
print(f"✅ Tier map saved:       {MODEL_PATH}tier_map.pkl")
print(f"✅ Clustered dataset saved: data/diabetic_data_clustered.csv")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print(f"Patients segmented into 3 clinical risk tiers:")
for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
    size = (df['risk_tier'] == tier).sum()
    rate = df[df['risk_tier'] == tier]['readmitted_binary'].mean() * 100
    print(f"  {tier}: {size:,} patients | Readmission Rate: {rate:.2f}%")