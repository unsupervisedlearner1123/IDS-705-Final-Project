"""Model Building Begins."""
## Import Required Packages
from preprocessing_pipeline import (
    shortlisted_features,
    cols_to_drop,
    categorical_cols,
    nulls_cols,
    corr_cols_drop,
    log_corr_cols_to_drop,
    cats_to_drop,
    pre_process_df,
    preprocessing_freq_encode,
)

from random import Random
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
import seaborn as sns
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import hyperopt
from hyperopt import Trials, fmin, tpe, hp
import matplotlib.pyplot as plt
from scipy.stats import loguniform


warnings.filterwarnings("ignore")
os.chdir("/add/your/path/here/")


## Import datasets
train = pd.read_csv("./data/train.csv")
train.name = "Train"
val = pd.read_csv("./data/val.csv")
val.name = "Validation"
test = pd.read_csv("./data/test.csv")
test.name = "Test"

# concat train and validation data
train_val = pd.concat([train, val])

## seperate features and response
y_train = train_val["HasDetections"]
train_features = train_val.drop(columns="HasDetections")

y_test = test["HasDetections"]
test_features = test.drop(columns="HasDetections")


## preprocess datasets to pass them in the model
preprocessing = preprocessing_freq_encode(10_000)
preprocessing.fit(train_features)
clean_train = preprocessing.transform(train_features)
clean_test = preprocessing.transform(test_features)


lgb_X_train = np.array(clean_train)
lgb_X_test = np.array(clean_test)
lgb_y_train = np.array(y_train)
lgb_y_test = np.array(y_test)

## initialize models
DC = DecisionTreeClassifier()

Rf = RandomForestClassifier(
    criterion="entropy",
    max_depth=24,
    max_features=50,
    n_estimators=40,
    random_state=42,
    n_jobs=-1,
)


XGB = XGBClassifier(
    colsample_bytree=0.23,
    max_depth=14,
    gamma=0.16,
    learning_rate=0.06,
    min_child_weight=5,
    n_estimators=115,
    subsample=0.14,
    random_state=42,
    nthread=-1,
)

lgb = LGBMClassifier(
    colsample_bytree=0.5,
    learning_rate=1.004603577028842,
    min_child_weight=6,
    n_estimators=134,
    num_leaves=14,
    subsample=0.5525456435860483,
    random_state=42,
    n_jobs=-1,
)


## fit_models
DC.fit(clean_train, y_train)
Rf.fit(clean_train, y_train)
XGB.fit(clean_train, y_train)
lgb.fit(lgb_X_train, lgb_y_train)

## make prediction on test data
DC_pred = DC.predict_proba(clean_test)[:, 1]
Rf_pred = Rf.predict_proba(clean_test)[:, 1]
XGB_pred = XGB.predict_proba(clean_test)[:, 1]
lgb_pred = lgb.predict_proba(lgb_X_test)[:, 1]


## make prediction on test data
DC_pred_label = DC.predict(clean_test)
Rf_pred_label = Rf.predict(clean_test)
XGB_pred_label = XGB.predict(clean_test)
lgb_pred_label = lgb.predict(lgb_X_test)

## calculate AUCs
DC_AUC = roc_auc_score(y_test, DC_pred)
Rf_AUC = roc_auc_score(y_test, Rf_pred)
XGB_AUC = roc_auc_score(y_test, XGB_pred)
LGB_AUC = roc_auc_score(y_test, lgb_pred)

## calculate AP
DC_AP = average_precision_score(y_test, DC_pred_label)
Rf_AP = average_precision_score(y_test, Rf_pred_label)
XGB_AP = average_precision_score(y_test, XGB_pred_label)
LGB_AP = average_precision_score(y_test, lgb_pred_label)


### save models
with open("./trained_models/DC_model.pkl", "wb") as f:
    pickle.dump(DC, f)

with open("./trained_models/Rf_model.pkl", "wb") as f:
    pickle.dump(Rf, f)

with open("./trained_models/XGB_model.pkl", "wb") as f:
    pickle.dump(XGB, f)

with open("./trained_models/lgb_model.pkl", "wb") as f:
    pickle.dump(lgb, f)

# plot ROC Curves
plt.figure(figsize=(7, 6))
models = [DC_pred, Rf_pred, XGB_pred, lgb_pred]
AUCS = {
    "Decision Tree": DC_AUC,
    "Random Forest": Rf_AUC,
    "XGBoost": XGB_AUC,
    "Light GBM": LGB_AUC,
}
plt.plot([0, 1], [0, 1], label="Random Guess AUC: 50%", linestyle="dashed")
for i, j in zip(models, AUCS.keys()):
    label = str(j) + " " + "AUC:" + " " + str(round(AUCS[j] * 100, 2)) + "%"
    fpr, tpr, _ = roc_curve(y_test, i)
    plt.plot(fpr, tpr, label=label)

plt.legend(loc="lower right", fontsize=13)
sns.despine()
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves", fontsize=17)

## Plot PR Curves
plt.figure(figsize=(7, 6))
models = [DC_pred_label, Rf_pred_label, XGB_pred_label, lgb_pred_label]

APS = {
    "Decision Tree": DC_AP,
    "Random Forest": Rf_AP,
    "XGBoost": XGB_AP,
    "Light GBM": LGB_AP,
}
AP_Chance = round(sum(test["HasDetections"]) / len(test["HasDetections"]), 2)

plt.plot(
    [0, 1], [AP_Chance, AP_Chance], label="Random Chance AP: 49.97%", linestyle="dashed"
)
for i, j in zip(models, APS.keys()):
    label = str(j) + " " + "AP:" + " " + str(round(APS[j] * 100, 2)) + "%"
    pre, rec, _ = precision_recall_curve(y_test, i)
    plt.plot(pre, rec, label=label)

plt.legend(loc="lower left", fontsize=13)
sns.despine()
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision Recall Curves", fontsize=17)


# find optimal cutoffs
def find_cutoff(x):
    """Find cut off that lead to best accuracy."""
    cut_offs = list(np.arange(0.1, 1, 0.01))
    accs = []
    probs = []
    for i in cut_offs:
        acc = accuracy_score(y_test, np.where(lgb_pred > i, 1, 0))
        probs.append(i)
        accs.append(acc)
    index = np.argmax(accs)
    return accs[index], probs[index]


## finding the best model with best cutoff
for i in models:
    print(find_cutoff(i))

"""0.49 as cutt off for all models"""

# best model Cf
cf_matrix = confusion_matrix(
    y_test,
    lgb.predict(
        lgb_X_test,
    ),
)

### plotting confusion matrix
group_names = [
    "True Negatives\n",
    "False Positives\n",
    "False Negatives\n",
    "True Positives\n",
]

plt.figure(figsize=(6, 5))
group_counts = ["{:,}".format(value) for value in cf_matrix.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]

labels = np.asarray(labels).reshape(2, 2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues", annot_kws={"size": 14})

ax.set_title("Light GBM Confusion Matrix\n\n", fontsize=17)
ax.set_xlabel("\nPredicted Values", fontsize=15)
ax.set_ylabel("Actual Values ", fontsize=15)

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(["False", "True"], fontsize=14)
ax.yaxis.set_ticklabels(["False", "True"], fontsize=14)

## Display the visualization of the Confusion Matrix.
plt.show()
