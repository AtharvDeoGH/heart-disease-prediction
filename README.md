Optimizing Heart Disease Prediction (Machine Learning)
This repo contains my end-to-end project using classic ML and gradient boosting to predict cardiovascular disease risk from clinical and lifestyle features. It covers EDA, feature engineering, model baselines, hyperparameter tuning, and a final LightGBM model selection. 
What's inside
Dataset: cardio_train.csv — ~70k rows × 13 columns of patient records (objective, examination, and subjective attributes).
Notebook: Python Project Group 3.ipynb — full workflow: EDA → preprocessing → modeling → evaluation.
Deck: Executive Summary - Heart Disease Prediction.pptx — business context, approach, and results for non-technical stakeholders.
Highlights
Goal: earlier, more accurate heart-disease risk prediction to support preventive care and reduce costs.
EDA: summary stats, correlation heatmaps, pair plots, and relationship checks to guide feature choices.
Feature engineering: BMI from height/weight, blood-pressure categories, cholesterol/glucose bands, and a combined lifestyle score.
Models tried: Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost, LightGBM.
Tuning: grid/random search over key hyperparameters (e.g., depth, estimators, learning rate, regularization).
Metrics: Accuracy, Precision, ROC-AUC (with ROC curves and confusion matrices).
Outcome: LightGBM selected for best overall performance/efficiency on this dataset.