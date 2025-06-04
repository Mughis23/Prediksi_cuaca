# -*- coding: utf-8 -*-


# ================================
# Section 1: Import Library
# ================================
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Section 2: Load dan Cek Kolom Data
# ================================
df = pd.read_csv('/content/data_cuaca.csv', sep=';')

print(df.head())
print("\nDaftar kolom:", df.columns.tolist())

# ================================
# Section 3: Buat Variabel Target Biner dari 'ch'
# ================================
df['rain'] = (df['ch'] > 0).astype(int)

print("Distribusi kelas 'rain':\n", df['rain'].value_counts())

# ================================
# Section 4: Pisahkan Fitur dan Target
# ================================
X = df.drop(columns=['rain', 'ch'])
y = df['rain']

# ================================
# Section 3.5: Imputasi Missing Values
# ================================
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ================================
# Section 5: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# Section 6: Baseline Random Forest
# ================================
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

y_pred_base = rf_baseline.predict(X_test)
y_proba_base = rf_baseline.predict_proba(X_test)[:, 1]

print("=== Baseline Random Forest ===")
print(classification_report(y_test, y_pred_base, digits=4))
auc_base = roc_auc_score(y_test, y_proba_base)
print(f"ROC-AUC (baseline): {auc_base:.4f}")

# ================================
# Section 7: Random Forest + SMOTE (Oversampling)
# ================================
smote = SMOTE(random_state=42)
pipeline_smote = ImbPipeline([
    ('smote', smote),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
pipeline_smote.fit(X_train, y_train)

y_pred_smote = pipeline_smote.predict(X_test)
y_proba_smote = pipeline_smote.predict_proba(X_test)[:, 1]

print("=== RF + SMOTE ===")
print(classification_report(y_test, y_pred_smote, digits=4))
auc_smote = roc_auc_score(y_test, y_proba_smote)
print(f"ROC-AUC (SMOTE): {auc_smote:.4f}")

# ================================
# Section 8: Random Forest + Undersampling
# ================================
rus = RandomUnderSampler(random_state=42)
pipeline_rus = ImbPipeline([
    ('undersample', rus),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
pipeline_rus.fit(X_train, y_train)

y_pred_rus = pipeline_rus.predict(X_test)
y_proba_rus = pipeline_rus.predict_proba(X_test)[:, 1]

print("=== RF + Undersampling ===")
print(classification_report(y_test, y_pred_rus, digits=4))
auc_rus = roc_auc_score(y_test, y_proba_rus)
print(f"ROC-AUC (Undersampling): {auc_rus:.4f}")

# ================================
# Section 9: Random Forest + Class Weight Balanced
# ================================
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_weighted.fit(X_train, y_train)

y_pred_weighted = rf_weighted.predict(X_test)
y_proba_weighted = rf_weighted.predict_proba(X_test)[:, 1]

print("=== RF + Class Weight ===")
print(classification_report(y_test, y_pred_weighted, digits=4))
auc_weighted = roc_auc_score(y_test, y_proba_weighted)
print(f"ROC-AUC (Class Weight): {auc_weighted:.4f}")

# ================================
# Section 10: Plot ROC Curve Perbandingan
# ================================
plt.figure(figsize=(8, 6))

fpr_base, tpr_base, _ = roc_curve(y_test, y_proba_base)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_proba_smote)
fpr_rus, tpr_rus, _ = roc_curve(y_test, y_proba_rus)
fpr_w, tpr_w, _ = roc_curve(y_test, y_proba_weighted)

plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC={auc_base:.3f})')
plt.plot(fpr_smote, tpr_smote, label=f'SMOTE (AUC={auc_smote:.3f})')
plt.plot(fpr_rus, tpr_rus, label=f'Undersample (AUC={auc_rus:.3f})')
plt.plot(fpr_w, tpr_w, label=f'Class Weight (AUC={auc_weighted:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Perbandingan Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ================================
# Section 11: Ringkasan ROC-AUC
# ================================
print("Ringkasan ROC-AUC:")
print(f"Baseline:      {auc_base:.4f}")
print(f"SMOTE:         {auc_smote:.4f}")
print(f"Undersampling: {auc_rus:.4f}")
print(f"Class Weight:  {auc_weighted:.4f}")
