import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.decomposition import PCA

#Load and clean datasets, filling empty with nan
df = pd.read_csv("UNSW_NB15_training-set.csv")
df.drop(['id'], axis=1, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encode columns for readibility
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Feature Scaling
X_train = df.drop(['label', 'attack_cat'], axis=1)
y_train = df['label']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------------- LOAD & CLEAN TEST DATA ---------------------- #
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
test_df.drop(['id'], axis=1, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.dropna(inplace=True)

# Encode test to match with training making tcp = 0, udp= 1, icmp=2
for col in cat_cols:
    if col in test_df.columns and col in label_encoders:
        le = label_encoders[col]
        test_df[col] = test_df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

X_test = test_df.drop(['label', 'attack_cat'], axis=1)
y_test = test_df['label']
X_test_scaled = scaler.transform(X_test)

# ---------------------- DECISION TREE ---------------------- #
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_scaled, y_train)
y_pred_dt = dt_clf.predict(X_test_scaled)
print("Decision Tree Report:")
print(classification_report(y_test, y_pred_dt))

# ---------------------- K-MEANS (ANOMALY DETECTION) ---------------------- #
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_scaled)
y_pred_kmeans = kmeans.predict(X_test_scaled)

def remap_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in range(2):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask], keepdims=True)[0]
    return labels

y_pred_kmeans_remapped = remap_labels(y_test.to_numpy(), y_pred_kmeans)
print("K-Means Report (Anomaly Detection):")
print(classification_report(y_test, y_pred_kmeans_remapped))
# Reduce test data to 2D for visualization

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# Plot clusters predicted by K-Means
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_kmeans_remapped, cmap='coolwarm', alpha=0.6, s=10)

plt.title("K-Means Clustering of Test Data (2D PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, ticks=[0, 1], label='Cluster Label (0=Normal, 1=Attack)')
plt.grid(True)
plt.show()

# ---------------------- RANDOM FOREST ---------------------- #
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)
print("Random Forest Report (Hybrid Detection):")
print(classification_report(y_test, y_pred_rf))

# ---------------------- FEATURE IMPORTANCE ---------------------- #
importances = rf_clf.feature_importances_
feature_names = X_train.columns

feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_n = 10
print(f"\nTop {top_n} Most Important Features (Random Forest):")
print(feat_importance_df.head(top_n))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_importance_df['Feature'][:top_n][::-1], feat_importance_df['Importance'][:top_n][::-1])
plt.xlabel("Importance Score")
plt.title(f"Top {top_n} Most Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# ---------------------- TRAIN RF ON TOP 10 FEATURES ---------------------- #
top10_features = feat_importance_df['Feature'].head(10).tolist()

X_train_top10 = df[top10_features]
X_test_top10 = test_df[top10_features]

scaler_top10 = StandardScaler()
X_train_top10_scaled = scaler_top10.fit_transform(X_train_top10)
X_test_top10_scaled = scaler_top10.transform(X_test_top10)

rf_top10 = RandomForestClassifier(random_state=42)
rf_top10.fit(X_train_top10_scaled, y_train)
y_pred_top10 = rf_top10.predict(X_test_top10_scaled)

print("\nRandom Forest with Top 10 Features:")
print(classification_report(y_test, y_pred_top10))

# ---------------------- ATTACK TYPE DISTRIBUTION ---------------------- #
attack_labels = label_encoders['attack_cat'].inverse_transform(df['attack_cat'])
attack_counts = pd.Series(attack_labels).value_counts()
print("\nAttack Type Distribution in Training Set:")
print(attack_counts)
