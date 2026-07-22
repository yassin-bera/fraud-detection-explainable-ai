
# Fraud Detection with Explainable AI

## Overview

A Machine Learning project for detecting fraudulent credit card transactions using supervised learning models and Explainable AI techniques.

The goal of this project is not only to predict fraudulent transactions but also to explain why the model makes each decision.

## Project Pipeline

Data Loading
↓
Data Preprocessing
↓
Train/Test Split
↓
Feature Scaling
↓
Handling Class Imbalance (SMOTE)
↓
Model Training
↓
Evaluation
↓
Explainability with SHAP


## Models Used

- Logistic Regression
- Logistic Regression + SMOTE
- XGBoost Classifier


## Results

Best Model: XGBoost

ROC-AUC Score: 0.9646

Fraud Detection Performance:

- Precision: 0.93
- Recall: 0.79
- F1-score: 0.85


## Explainable AI

SHAP (SHapley Additive exPlanations) was used to understand model decisions.

The project provides:

- Global Feature Importance
- SHAP Summary Plot
- Local Explanation for Individual Transactions


## Project Structure
