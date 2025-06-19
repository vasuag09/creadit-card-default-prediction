# Credit Card Default Prediction 🧾🔍

Predicting whether a credit card customer will default in the next month using classification models.

---

## 📊 Problem Statement

This project aims to develop a predictive model that classifies whether a customer will default on their credit card payment next month. The goal is to help financial institutions reduce risk by identifying high-risk customers early.

---

## 📁 Dataset

**Source**: [UCI Default of Credit Card Clients - Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

- 30,000 records
- 23 features + 1 target
- Target variable: `default.payment.next.month` (1 = default, 0 = no default)
- No missing values
- Some categorical features already encoded numerically

---

## 🧹 Preprocessing Steps

- Removed `ID` column
- Split into `X` (features) and `y` (target)
- Converted features and target to NumPy arrays
- Applied:
  - **One-Hot Encoding** on categorical columns: `SEX`, `EDUCATION`, `MARRIAGE`
  - **Standard Scaling** on numerical columns
- Concatenated encoded + scaled data into final `X`

---

## 🧠 Models Used

| Model               | Accuracy | Class 1 Recall | Notes                         |
|--------------------|----------|----------------|-------------------------------|
| Logistic Regression | ~78.4%   | ~0.07%         | Poor at detecting defaulters  |
| Random Forest       | ~82.2%   | ~37%           | Balanced performance with `class_weight='balanced'` |

---

## 📈 Best Model (Random Forest)

- **Classifier**: `RandomForestClassifier(n_estimators=55, criterion='entropy', class_weight='balanced')`
- **Accuracy**: 82.2%
- **Recall for Defaulters (Class 1)**: 37%
- **F1-Score for Defaulters**: 0.48

Confusion Matrix:
[[4448 255]
[ 812 485]]

## 📌 Limitations

- Dataset is **highly imbalanced** (only ~22% defaulters)
- Model still misses many defaulters (false negatives)
- Future improvements:
  - Apply **threshold tuning**
  - Use **SMOTE** for oversampling
  - Try **XGBoost**, **ensemble methods**, or **PCA** for feature reduction
