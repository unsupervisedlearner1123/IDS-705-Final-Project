"""Experiments and Simulations"""
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

## Import Required Packages
import numpy as np
import pandas as pd
import pickle
import os
import warnings

os.chdir("/add/your/path/here/")
warnings.filterwarnings("ignore")

## set seed in numpy to replicate
np.random.seed(145)

## Import datasets
train = pd.read_csv("./data/train.csv")
train_features = train.drop(columns="HasDetections")

# import model
lgb = pickle.load(open("./trained_models/lgb_model.pkl", "rb"))

## take a subset of data with 5000 features
subset_data = train.loc[np.random.choice(train.index, size=200000)]
subset_y = np.array(subset_data["HasDetections"].copy())
subset_X = subset_data.drop(columns="HasDetections")


test_col = list(
    [
        "Census_ProcessorCoreCount",
        "Census_SystemVolumeTotalCapacity",
        "Census_InternalPrimaryDiagonalDisplaySizeInInches",
        "Census_InternalPrimaryDisplayResolutionHorizontal",
        "Census_TotalPhysicalRAM",
    ]
)

test_quantile = list([0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9])
process_data = preprocessing_freq_encode(10_000)
process_data.fit(train_features)
simulated_probs = {}
quantile_dict = {}
for i in test_col:
    pred_probs = []
    quantiles = []
    for j in test_quantile:
        name = i + " " + str(j * 100) + " percentile"
        test2 = subset_data.copy()
        test2[i] = train[i].quantile(j)
        quantiles.append(train[i].quantile(j))
        data = np.array(process_data.transform(test2))
        pred = round(np.mean(lgb.predict(data)) * 100, 2)
        pred_prob = round(np.mean(lgb.predict_proba(data)[:, 1]), 2)
        pred_probs.append(pred_prob)

    simulated_probs[i] = pred_probs
    quantile_dict[i] = quantiles

quantile_data = pd.DataFrame(quantile_dict)
probability_data = pd.DataFrame(simulated_probs)

quantile_data.to_csv("./results/Quantiles_data.csv")
probability_data.to_csv("./results/Pred_probs.csv")
