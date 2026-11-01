# Telecom Customer Churn Prediction - Classification Models + Streamlit App

Machine Learning, Assignment 2

Student : UMANG GUPTA 

BITSID : 2026AK05537





## a. Problem statement

Customer churn (a subscriber leaving the operator) is costly for telecom companies. The goal is to
predict whether a customer will **churn (1)** or **stay (0)** from their usage and account
attributes. Five classifiers - Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian
Naive Bayes and Support Vector Machine - are trained on the same dataset, compared with six
metrics (Accuracy, AUC, Precision, Recall, F1, MCC) and exposed through a deployed Streamlit app.







## b. Dataset description

### Source

* **Name:** Iranian Churn Dataset
* **Repository:** UCI Machine Learning Repository (dataset ID 563) - https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
* **Direct download:** https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip (contains `Customer Churn.csv`, \~128 KB)
* **Donated:** 8 April 2020 - **Subject area:** Business - **Tasks:** Classification, Regression - **Feature type:** Integer
* **Origin:** randomly sampled from the customer database of an Iranian telecom company; originally used in
Jafari-Marandi, R., Denton, J., Idris, A., Smith, B. K., \& Keramati, A. (2020), *Optimum Profit-Driven Churn
Decision Making: Innovative Artificial Neural Networks in Telecom Industry*, Neural Computing and Applications.
* **License:** Creative Commons Attribution 4.0 (CC BY 4.0)
* **Citation:** Iranian Churn \[Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5JW3Z

```python
# how the data can be fetched programmatically
import pandas as pd
url = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
df = pd.read\_csv(url, compression="zip")        # or: from ucimlrepo import fetch\_ucirepo; fetch\_ucirepo(id=563)
```

### Size and collection window

|Property|Value|
|-|-|
|Instances (customers)|3,150 (one row per customer)|
|Input features|13 (assignment minimum: 12)|
|Target|`Churn` (binary: 1 = churned, 0 = stayed)|
|Missing values|None|
|Class balance|2,655 non-churn (84.3%) / 495 churn (15.7%) - imbalanced|
|Time frame|12 months|

All input features are **aggregated over the first 9 months**; the `Churn` label is the customer's state at the
**end of 12 months**. The 3 months in between are a designated *planning gap*, so the model predicts churn
with a realistic lead time instead of looking at data from after the customer left.

### Feature dictionary

|#|Feature|Type|Description|Range in data|
|-|-|-|-|-|
|1|Call Failure|Integer|Number of call failures|0 - 36|
|2|Complains|Binary|0 = no complaint, 1 = complaint|0 - 1|
|3|Subscription Length|Integer|Total months of subscription|3 - 47|
|4|Charge Amount|Ordinal|Billing amount level, 0 = lowest|0 - 10|
|5|Seconds of Use|Integer|Total seconds of calls|0 - 17,090|
|6|Frequency of use|Integer|Total number of calls|0 - 255|
|7|Frequency of SMS|Integer|Total number of text messages|0 - 522|
|8|Distinct Called Numbers|Integer|Number of distinct phone numbers called|0 - 97|
|9|Age Group|Ordinal|1 = youngest ... 5 = oldest bracket|1 - 5|
|10|Tariff Plan|Binary|1 = pay-as-you-go, 2 = contractual|1 - 2|
|11|Status|Binary|1 = active, 2 = non-active|1 - 2|
|12|Age|Integer|Age of the customer (years)|15 - 55|
|13|Customer Value|Continuous|Calculated value of the customer|0 - 2,165.28|
|-|**Churn** (target)|Binary|1 = churn, 0 = non-churn|0 - 1|

Customer IDs are anonymous and not included. (The UCI description lists `Charge Amount` as 0-9; the CSV
contains a maximum of 10.)



### Exploratory observations (computed on the full data)

* **Strongest churn signals:** customers who **complained** churn at 83.0% vs 10.1% for those who did not;
**non-active** customers (Status = 2) churn at 47.3% vs 5.3% for active ones. Correlation with churn:
Complains +0.53, Status +0.50.
* **Usage is protective:** Frequency of use (-0.30), Seconds of Use (-0.30), Customer Value (-0.29) and
Distinct Called Numbers (-0.28) are all negatively correlated with churn - low-usage customers leave.
* **Tariff plan:** contractual customers (plan 2) churn only 2.4% vs 16.8% for pay-as-you-go (plan 1).
* **Weak features:** Call Failure, Age, Age Group and Subscription Length show almost no linear correlation with churn.
* **Redundant features (multicollinearity):** Seconds of Use vs Frequency of use r = 0.95; Age vs Age Group
r = 0.96. This matters for Naive Bayes, which assumes independent features.
* **Scale differences:** Seconds of Use goes to 17,090 while Complains is 0/1, so features are standardised
(`StandardScaler`) for Logistic Regression, KNN, Naive Bayes and SVM.



### Data quality notes

* No missing values; all columns are numeric, so no encoding was needed.
* The file contains **300 exact duplicate rows**. They were kept because the dataset is anonymous
and identical usage profiles can legitimately belong to different customers. As a consequence, 95 of the 630
test rows also occur in the training set. To make sure this does not inflate the results, all models were
re-evaluated on the 535 test rows that do not appear in training: KNN was still best (Accuracy 0.957,
F1 0.857, MCC 0.832), so the ranking and the winner are unchanged.
* The target is imbalanced (15.7% churn), so accuracy alone is misleading; AUC, Recall, F1 and MCC are reported too.



### Pre-processing and split

Stratified 80/20 train/test split (`random\_state=42`) -> 2,520 training and 630 test rows, with the same
churn ratio in both. `StandardScaler` is part of each model pipeline (not fitted on test data), 

so the saved `.joblib` models are self-contained. The 630-row held-out set is saved as `test\_data.csv` and is what is
uploaded to the Streamlit app.



## c. GitHub repository link

https://github.com/ak05537/ml\_assignment

```
ml\_assignment/
|-- streamlit\_app.py              # Streamlit app
|-- common.py           # shared metric helpers
|-- train.py            # trains + saves the 5 models, writes test\_data.csv
|-- requirements.txt
|-- README.md
|-- test\_data.csv       # held-out test data (features + Churn)
|-- data/Customer\_Churn.csv   # full original dataset
`-- model/              # saved pipelines: \*.joblib, metrics.csv, feature\_names.json
```

**Run locally**

```bash
pip install -r requirements.txt
python train.py          # optional - regenerates model/ and test\_data.csv
streamlit run app.py
```

## d. Models used

Evaluation on the 630-row held-out test set (churn = positive class):

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-|-|-|-|-|-|
|Logistic Regression|0.8413|0.9243|0.4971|0.8586|0.6296|0.5701|
|Decision Tree|0.8540|0.9310|0.5212|0.8687|0.6515|0.5959|
|KNN|**0.9587**|0.9578|**0.8925**|0.8384|**0.8646**|**0.8408**|
|Naive Bayes|0.7381|0.8986|0.3648|0.8990|0.5190|0.4536|
|Support Vector Machine|0.8571|**0.9647**|0.5263|**0.9091**|0.6667|0.6192|





**Hyper-parameters:**

&#x20;LR (`max\_iter=1000`, `class\_weight=balanced`);

&#x20;Decision Tree (`max\_depth=6`,`min\_samples\_leaf=10`, `class\_weight=balanced`); 

KNN (`k=7`, distance-weighted, scaled); 

Gaussian NB (defaults, scaled); 

SVM (RBF, `C=1`, `class\_weight=balanced`, `probability=True`).





### Observations on model performance

|ML Model Name|Observation about model performance|
|-|-|
|Logistic Regression|Good ranking ability (AUC 0.92) and high recall (0.86), but precision is only \~0.50: with `class\_weight=balanced` it flags many loyal customers as churners (accuracy 0.84). A linear boundary cannot capture the non-linear usage patterns.|
|Decision Tree|Slightly better than LR on every metric (F1 0.65, MCC 0.60) and fully interpretable, but depth was limited to 6 to avoid over-fitting, so it still trades precision (0.52) for recall (0.87).|
|KNN|Best overall. Once features are scaled, similar customers in usage-space share churn behaviour; it achieves precision 0.89 and recall 0.84 together, giving F1 0.86 and MCC 0.84. It is the only model that is not over-predicting churn.|
|Naive Bayes|Weakest model: accuracy 0.74, precision 0.36, MCC 0.45. Features such as Seconds of Use, Frequency of use and Customer Value are strongly correlated, violating the independence assumption. It has high recall (0.90) but at the cost of many false alarms.|
|Support Vector Machine|Best AUC (0.965) and best recall (0.91) - the best at ranking and catching churners - but with balanced class weights its precision is \~0.53, so F1/MCC are well below KNN at the default 0.5 threshold.|
|**Overall winner for this dataset**|**KNN** - highest Accuracy, Precision, F1 and MCC (MCC is the most reliable single number on this imbalanced data). SVM is the runner-up and would be preferred if maximising recall/AUC matters more than precision.|

## 

## Streamlit app features

1. Upload a CSV (test data only - needs the 13 feature columns + `Churn`).
2. Dropdown to choose any of the 5 saved models.
3. Accuracy, AUC, Precision, Recall, F1 and MCC displayed.
4. Confusion matrix, ROC curve and classification report; comparison table of all models on the uploaded data; downloadable predictions.
5. 

Models are loaded from `model/` - **nothing is retrained** on the uploaded file.



**Git Hub Repository:** https://github.com/ak05537/ml\_assignment

**Live app:** https://mlassignment-545wotbk92hhhth9hqqkj6.streamlit.app/ 

