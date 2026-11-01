"""Streamlit app: Telecom Customer Churn - 5 classifier comparison.

Loads the pre-trained models from model/ (NO retraining), scores an uploaded
test CSV, and shows metrics, confusion matrix and classification report.
"""
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve

from common import TARGET, MODEL_FILES, clean_columns, compute_metrics, positive_scores

st.set_page_config(page_title="Telecom Churn Classifier Lab", page_icon="📉", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, "test_data.csv"))


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Controls")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
uploaded = st.sidebar.file_uploader("Upload TEST data (CSV)", type="csv")
st.sidebar.caption(f"The CSV must contain all 13 feature columns plus the target column "
                   f"**{TARGET}** (0 = stays, 1 = churns).")
st.sidebar.download_button("⬇️ Download sample test_data.csv",
                           load_sample().to_csv(index=False).encode(),
                           file_name="test_data.csv", mime="text/csv")

# ---------------------------------------------------------------- header
st.title("📉 Telecom Customer Churn - Model Comparison")
st.write("Iranian telecom dataset · 5 classifiers trained offline "
         "(Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM). "
         "Upload test data to evaluate any saved model - nothing is retrained here.")

if uploaded is None:
    st.info("👈 Upload a test CSV in the sidebar (or download the sample and re-upload it).")
    st.subheader("Training-time results on the held-out test set")
    st.dataframe(pd.read_csv(os.path.join(BASE, "model", "metrics.csv")).set_index("ML Model Name"),
                 use_container_width=True)
    st.stop()

# ---------------------------------------------------------------- validate
df = clean_columns(pd.read_csv(uploaded))
model = load_model(model_name)
required = list(model.feature_names_in_)
missing = [c for c in required + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required column(s): {missing}")
    st.stop()
if df[required + [TARGET]].isna().any().any():
    st.error("The CSV contains missing values - please clean it first.")
    st.stop()
if df[TARGET].nunique() < 2:
    st.error("Target column needs both classes (0 and 1) to compute AUC / MCC.")
    st.stop()

X, y = df[required], df[TARGET].astype(int)
y_pred = model.predict(X)
y_score = positive_scores(model, X)
metrics = compute_metrics(y, y_pred, y_score)

st.success(f"Evaluated **{model_name}** on **{len(df)}** rows.")

# ---------------------------------------------------------------- metrics
st.subheader(f"📊 Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(["Confusion matrix", "Classification report",
                                  "All models compared", "Predictions"])

with tab1:
    c1, c2 = st.columns(2)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    c1.pyplot(fig)
    fpr, tpr, _ = roc_curve(y, y_score)
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC curve"); ax2.legend()
    c2.pyplot(fig2)

with tab2:
    rep = classification_report(y, y_pred, target_names=["Stay (0)", "Churn (1)"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

with tab3:
    rows = []
    for name in MODEL_FILES:
        m = load_model(name)
        rows.append({"ML Model Name": name,
                     **compute_metrics(y, m.predict(df[list(m.feature_names_in_)]),
                                       positive_scores(m, df[list(m.feature_names_in_)]))})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"),
                 use_container_width=True)
    st.bar_chart(comp[["Accuracy", "AUC", "F1", "MCC"]])

with tab4:
    out = df.copy()
    out["Predicted"] = y_pred
    out["Churn_probability"] = y_score.round(4)
    st.dataframe(out, use_container_width=True)
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")
