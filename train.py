"""Train the 5 classifiers on the Iranian Telecom churn dataset and save them.

Usage:  python train.py
Outputs: model/*.joblib, model/metrics.csv, test_data.csv
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from common import TARGET, MODEL_FILES, clean_columns, compute_metrics, positive_scores

RANDOM_STATE = 42
DATA_PATH = "data/Customer_Churn.csv"

df = clean_columns(pd.read_csv(DATA_PATH))
X, y = df.drop(columns=[TARGET]), df[TARGET]

# Stratified 80/20 split keeps the ~16% churn ratio in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

# Scaling lives inside each Pipeline, so the saved .joblib is self-contained
# and the app never needs to re-fit anything on uploaded data.
models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE))]),
    "Decision Tree": Pipeline([  # trees are scale-invariant: no scaler
        ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE))]),
    "KNN": Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
    "Naive Bayes": Pipeline([
        ("scale", StandardScaler()),
        ("clf", GaussianNB())]),
    "Support Vector Machine": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, class_weight="balanced",
                    probability=True, random_state=RANDOM_STATE))]),
}

rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, f"model/{MODEL_FILES[name]}")
    m = compute_metrics(y_test, pipe.predict(X_test), positive_scores(pipe, X_test))
    rows.append({"ML Model Name": name, **m})
    print(f"{name:24s}", {k: round(v, 4) for k, v in m.items()})

pd.DataFrame(rows).round(4).to_csv("model/metrics.csv", index=False)
json.dump(list(X.columns), open("model/feature_names.json", "w"))

# Held-out test set (features + target) -> upload this in the Streamlit app
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)
print(f"\nTrain rows: {len(X_train)}  Test rows: {len(X_test)}  -> test_data.csv saved")
