# Fetal Health Classification from Cardiotocogram (CTG) Data

This project builds and evaluates machine learning models to **predict fetal health status** (Normal / Suspect / Pathological) from **Cardiotocogram (CTG)** data.

The goal is to support clinicians with a data-driven tool that can flag high-risk fetal conditions early using features derived from CTG measurements.

---

## 🔍 Problem Overview

Fetal distress and complications during pregnancy can lead to severe outcomes if not detected early. CTG monitoring generates multiple time-series and statistical features related to fetal heart rate and uterine contractions.

In this project, we:

- Load and clean the **fetal_health** dataset  
- Perform **EDA and visualization** to understand feature relationships  
- Build and compare multiple ML models:
  - 🔹 Random Forest  
  - 🔹 Logistic Regression (multinomial)  
  - 🔹 XGBoost  
- Evaluate models using:
  - Accuracy, F1-score  
  - ROC-AUC  
  - Confusion Matrix  
- Apply **PCA** for 2D visualization of the feature space

---

## 📂 Project Structure

```text
.
├── fetal_health-1(5).csv         # Fetal health dataset (CTG-based features)
├── fetal_health_classification.py # Main script (rename from original)
├── DM_REPORT1_DHRUVPATEL.pdf     # Project report (methodology & results)
└── README.md                     # Project documentation
