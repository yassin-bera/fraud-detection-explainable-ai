# Fraud Detection with Explainable AI

A Machine Learning project for detecting fraudulent credit card transactions using **XGBoost** and **Explainable AI (SHAP)** techniques.

The goal of this project is not only to predict fraudulent transactions but also to explain **why** the model makes each decision.

---

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions are extremely rare compared to normal transactions.

This project implements a complete ML pipeline:

- Data loading and exploration
- Data preprocessing
- Feature scaling
- Handling class imbalance
- Model training
- Model evaluation
- Explainable AI analysis
- Transaction-level prediction

---

# Machine Learning Pipeline

```
Data Collection
        |
        ↓
Data Preprocessing
        |
        ↓
Train/Test Split
        |
        ↓
Feature Scaling
        |
        ↓
Class Imbalance Handling
        |
        ↓
Model Training
        |
        ↓
Performance Evaluation
        |
        ↓
Explainable AI with SHAP
```

---

# Dataset

Dataset:

**Credit Card Fraud Detection Dataset**

Source:
Kaggle / European cardholder transactions dataset

Dataset characteristics:

- Total transactions: 284,807
- Features: 30
- Target variable: Class
- Fraud cases: highly imbalanced minority class

Target:

```
Class = 0 → Normal Transaction

Class = 1 → Fraudulent Transaction
```

---

# Models Implemented

## 1. Logistic Regression

Baseline classification model.

Performance:

- ROC-AUC: 0.9605


---

## 2. Logistic Regression + SMOTE

To handle class imbalance, Synthetic Minority Oversampling Technique (SMOTE) was applied.

Performance:

- ROC-AUC: 0.9708


---

## 3. XGBoost Classifier 

Final selected model.

Why XGBoost?

- High performance on tabular data
- Handles complex patterns
- Provides feature importance
- Compatible with SHAP explanations


Performance:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.9646 |
| Precision | 0.93 |
| Recall | 0.79 |
| F1-score | 0.85 |


---

# Explainable AI (XAI)

To understand model decisions, this project uses:

## SHAP
(**SHapley Additive exPlanations**)

SHAP explains the contribution of each feature to the prediction.

Implemented explanations:

- Global feature importance
- SHAP summary plot
- Individual transaction explanation


Example:

The model can explain:

```
Transaction Prediction: Fraud

Fraud Probability:
99.5%

Main contributing features:
- V14
- V10
- V12
- V17
```

---

# Project Structure

```
fraud-detection-explainable-ai/

│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_xgb_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── fraud_detection_project.ipynb
│
├── src/
│   └── predict.py
│
├── requirements.txt
│
└── README.md
```

---

# Prediction Example

The trained model can predict new transactions.

Example:

```python
from predict import predict_transaction

result = predict_transaction(transaction)

print(result)
```

Output:

```json
{
    "prediction": 0,
    "fraud_probability": 0.0000015
}
```

Prediction:

```
0 → Normal Transaction

1 → Fraudulent Transaction
```

---

# Technologies Used

Programming Language:

- Python


Libraries:

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Seaborn


---

# Installation

Clone the repository:

```bash
git clone https://github.com/yassin-bera/fraud-detection-explainable-ai.git
```

Move into project directory:

```bash
cd fraud-detection-explainable-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Prediction

Run:

```bash
python src/predict.py
```

---

# Future Improvements

Possible improvements:

- Deploy model using FastAPI
- Create Streamlit dashboard
- Add real-time fraud detection API
- Add model monitoring
- Add Docker support
- Improve explainability dashboard


---

# Author

**Yassin Bera**

Machine Learning Project  
Fraud Detection + Explainable AI


---

# License

This project is for educational and research purposes.
  ## Model Explainability Visualizations

### Feature Importance

The following plot shows the most important features used by the XGBoost model:

![Feature Importance](images/feature_importance.png)


### SHAP Summary Plot

SHAP summary plot explains how each feature contributes to fraud predictions globally:

![SHAP Summary](images/shap_summary.png)


### SHAP Local Explanation

SHAP force plot explains why the model classified a specific transaction as fraudulent:

![SHAP Force Plot](images/shap_force_plot.png)
