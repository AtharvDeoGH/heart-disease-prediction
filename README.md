# Heart Disease Risk Prediction (LightGBM)

Predict the risk of cardiovascular disease from routine health indicators and lifestyle variables. This repo includes a dataset, a research notebook, a project summary deck, and environment requirements.

## Overview
- **Goal:** Earlier, more reliable heart disease risk prediction to support proactive care.
- **Approach:** Start with interpretable baselines (Logistic Regression, Decision Tree, KNN), then advance to ensembles (Random Forest, XGBoost, LightGBM). Final model selection favors **LightGBM** based on efficiency and overall metrics.
- **References:** See the executive deck in [`reports/Executive Summary.pptx`](reports/Executive%20Summary.pptx).

## Data
- **File:** `data/cardio_train.csv`
- **Shape observed:** `70,000` rows × 24 columns (after feature engineering).
- **Target column:** `cardio` (binary: 1 = disease, 0 = no disease).
- **Notes:** The original cardio dataset contains ~70k rows and ~13 core medical attributes; this project augments it with engineered features (BMI, blood pressure categories, cholesterol/glucose categories, and a combined lifestyle score).

> Tip: If your local filename includes a colon (e.g., `data:cardio_train.csv`), rename it to `data/cardio_train.csv` in your repo.

## Environment
Create a virtual environment and install the requirements. If you maintain dependencies in an RTF, convert it to a `requirements.txt` first (same package list).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# If you have requirements.txt:
pip install -r requirements.txt

# Or install essentials directly:
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn jupyter
```

## Project structure
```
.
├─ data/
│  └─ cardio_train.csv
├─ notebooks/
│  └─ heart_disease_prediction.ipynb
├─ reports/
│  └─ Executive Summary.pptx
├─ src/
│  ├─ train.py
│  └─ predict.py
└─ README.md
```

## Quickstart

### 1) Train (LightGBM example)

```python
# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

df = pd.read_csv("data/cardio_train.csv")
y = df["cardio"].astype(int)
X = df.drop(columns=["cardio"])

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_valid = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
}

bst = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_valid],
    valid_names=["valid"],
    early_stopping_rounds=50,
    verbose_eval=100,
)

proba = bst.predict(X_va, num_iteration=bst.best_iteration)
auc = roc_auc_score(y_va, proba)
print("Validation AUC:", round(auc, 4))
bst.save_model("lgbm_heart.txt")
```

### 2) Inference

```python
# src/predict.py
import pandas as pd
import lightgbm as lgb

bst = lgb.Booster(model_file="lgbm_heart.txt")
X_new = pd.read_csv("data/cardio_train.csv").drop(columns=["cardio"])  # replace with new data
proba = bst.predict(X_new, num_iteration=bst.best_iteration)
pd.DataFrame({"risk_probability": proba}).to_csv("predictions.csv", index=False)
print("Wrote predictions.csv")

## Modeling notes
- **Baselines:** Logistic Regression (probabilistic, interpretable), Decision Tree (non‑linear, visual), KNN (non‑parametric).
- **Ensembles:** Random Forest (reduced variance), XGBoost (regularized boosting), **LightGBM** (fast, accurate; strong candidate for production).
- **Feature engineering:** BMI from height/weight; blood pressure, cholesterol, and glucose category flags; combined lifestyle/behavior score.
- **Tuning ideas:** Adjust tree depth/leaves, learning rate, subsampling, and regularization; monitor ROC‑AUC and precision/recall.