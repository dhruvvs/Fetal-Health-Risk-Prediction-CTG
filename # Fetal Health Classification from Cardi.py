# Fetal Health Classification from Cardiotocogram Data

import pandas as pd # type: ignore
import numpy as np # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, precision_recall_curve, auc # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.decomposition import PCA # type: ignore
from scipy.stats import pearsonr # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from xgboost import XGBClassifier # type: ignore

# Step 1: Load the data
df = pd.read_csv(r"C:\Users\HP\Downloads\fetal_health-1(5).csv")
print(df.info())
print(df.head())

# Step 2: Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='fetal_health', data=df)
plt.title("Fetal Health Class Distribution")
plt.xlabel("Class (1=Normal, 2=Suspect, 3=Pathological)")
plt.ylabel("Count")
plt.show()

# Step 3: Feature correlation analysis
print("Calculating feature correlations with fetal health...")
df_clean = df.dropna()
significant_features = []
for col in df_clean.columns:
    if col != 'fetal_health' and pd.api.types.is_numeric_dtype(df_clean[col]):
        corr, pval = pearsonr(df_clean[col], df_clean['fetal_health'])
        if pval < 0.1:
            significant_features.append((col, corr, pval))

top_10_features = sorted(significant_features, key=lambda x: abs(x[1]), reverse=True)[:10]
print(pd.DataFrame(top_10_features, columns=["Feature", "Correlation", "p-value"]))

# Step 4: Data preparation
X = df[[x[0] for x in top_10_features]].dropna()
y = df['fetal_health'].loc[X.index].astype(int) - 1  # Convert to 0-based labels for XGBoost

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Step 5: Apply SMOTE
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Step 7: Train models
print("Training models on SMOTE-balanced data...")

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)

# Logistic Regression
lr = LogisticRegression(max_iter=2000, multi_class='multinomial')
lr.fit(X_train_res, y_train_res)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_res, y_train_res)

# Step 8: Confusion Matrices
print("Generating confusion matrices...")
models = [(rf, "Random Forest"), (lr, "Logistic Regression"), (xgb, "XGBoost")]
for model, name in models:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Suspect", "Pathological"])
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Step 9: Evaluation Metrics
print("Evaluating models...")

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovo')
    f1 = f1_score(y_test, y_pred, average='weighted')

    pr_aucs = []
    for i in range(3):
        precision, recall, _ = precision_recall_curve((y_test == i).astype(int), y_prob[:, i])
        pr_aucs.append(auc(recall, precision))
    pr_auc = np.mean(pr_aucs)

    print(f"{name} - ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}, PR AUC: {pr_auc:.4f}")

for model, name in models:
    evaluate_model(model, name)

# Step 10: K-means Clustering (unsupervised)
print("Performing K-Means clustering...")
X_full = df.drop(columns=['fetal_health']).select_dtypes(include=[np.number]).dropna()
X_pca = PCA(n_components=2).fit_transform(X_full)

for k in [5, 10, 15]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_full)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
