"""Shared helpers used by train.py and app.py (single source of truth for metrics)."""
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Churn"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The raw CSV has double spaces in headers ('Call  Failure'); collapse them."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    return df


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Six metrics required by the assignment (binary churn problem)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def positive_scores(model, X):
    """Probability of class 1 (all five pipelines expose predict_proba)."""
    return model.predict_proba(X)[:, 1]


def explain_winner(comp: pd.DataFrame, churn_rate: float) -> str:
    """Plain-language reasoning for the model ranking (markdown), built from live results."""
    ranked = comp.sort_values("MCC", ascending=False)
    best, runner = ranked.index[0], ranked.index[1]
    b = comp.loc[best]
    lines = [
        f"**Why {best} comes out on top for this upload**",
        f"- Only {churn_rate:.1%} of customers churn, so a model that predicts *'stays'* for everybody "
        f"would already score {1 - churn_rate:.1%} accuracy while finding no churners. "
        f"Accuracy alone therefore cannot pick the winner.",
        f"- MCC is used as the main yardstick because it combines true/false positives *and* negatives "
        f"and stays low unless both classes are handled well (0 = guessing, 1 = perfect).",
        f"- {best}: MCC {b['MCC']:.3f}, F1 {b['F1']:.3f}. It finds {b['Recall']:.0%} of the churners and "
        f"{b['Precision']:.0%} of the customers it flags really do leave. "
        f"Runner-up: {runner} (MCC {ranked.loc[runner, 'MCC']:.3f}).",
    ]
    top_recall = comp["Recall"].idxmax()
    if top_recall != best:
        lines.append(
            f"- Trade-off: {top_recall} has the highest recall ({comp.loc[top_recall, 'Recall']:.0%}) "
            f"but only {comp.loc[top_recall, 'Precision']:.0%} precision, i.e. many retention offers "
            f"would go to customers who were never going to leave.")
    return "\n".join(lines)
