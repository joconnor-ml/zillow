import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from zillow import modelling
import pickle as pkl
import dask.dataframe as dd
import sys

i = sys.argv[1]

test_df = pd.read_hdf("input/test2.{}.hdf".format(i), "data")
parcelid = test_df["ParcelId"]
date = test_df["transactiondate"]
month = test_df["yearmonth"]

with open("input/feat_names.pkl", "rb") as f:
    feat_names = pkl.load(f)

print(feat_names)
test_df = test_df[list(feat_names)]

for c in test_df.columns:
    if test_df[c].dtype == 'object':
        test_df[c] = LabelEncoder().fit_transform(list(test_df[c].values))

preds = {}
for name, model in modelling.stage1_models.items():
    with open("models/{}.py".format(name), "rb") as f:
        model = pkl.load(f)
    print(name)
    preds[name] = model.predict(test_df)

del test_df

test_df = pd.DataFrame(preds)
test_df.to_csv("stack_stage1_test.csv")

preds = {}
for name, model in modelling.stage2_models.items():
    with open("models/{}.py".format(name), "rb") as f:
        model = pkl.load(f)
    print(name)
    preds[name] = model.predict(test_df)

preds = pd.DataFrame(preds)
preds["ParcelId"] = parcelid
preds["date"] = date
preds.to_csv("preds_raw.{}.csv").format(i)