# bank_marketing_analysis.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# -----------------------------
# Precision@k and Recall@k helper
# -----------------------------
def pr_at_k(y_true, scores, k=0.2):
    n = len(scores)
    cutoff = int(np.ceil(k * n))
    idx = np.argsort(-scores)[:cutoff]   # top k%
    
    preds = np.zeros_like(y_true)
    preds[idx] = 1
    
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    return precision, recall, cutoff

# -----------------------------
# Load Data
# -----------------------------
# df = pd.read_csv("bank.csv", sep=";")
df = pd.read_csv("bank-full.csv", sep=";")


# Target: convert 'yes'/'no' to 1/0
df["y"] = df["y"].map({"yes": 1, "no": 0})

# Define features
numeric = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical = ["job","marital","education","default","housing","loan",
               "contact","month","poutcome"]

X = df[numeric + categorical]
y = df["y"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Preprocessing
# -----------------------------
prep = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical)
])

# -----------------------------
# Baseline Model: Logistic Regression
# -----------------------------
logreg = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000))
])
logreg.fit(X_train, y_train)
log_p = logreg.predict_proba(X_test)[:,1]

# -----------------------------
# Improved Model: Random Forest
# -----------------------------
rf = Pipeline([
    ("prep", prep),
    ("clf", RandomForestClassifier(n_estimators=250, random_state=42))
])
rf.fit(X_train, y_train)
rf_p = rf.predict_proba(X_test)[:,1]

# -----------------------------
# Evaluation
# -----------------------------
models = {"Logistic Regression": log_p, "Random Forest": rf_p}

for name, preds in models.items():
    auc = roc_auc_score(y_test, preds)
    print(f"\n{name} – ROC-AUC: {auc:.3f}")
    for k in [0.1, 0.2, 0.3]:
        p, r, n = pr_at_k(y_test.values, preds, k)
        print(f" Top {int(k*100)}% ({n} customers): Precision={p:.3f}, Recall={r:.3f}")
