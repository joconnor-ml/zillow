import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from zillow import modelling

import pickle as pkl


test_df = pd.read_hdf("input/test.hdf", "data")
parcelid = test_df["ParcelId"]
date = test_df["transactiondate"]

cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
pid = test_df["ParcelId"]
test_df = test_df.drop(['ParcelId', "parcelid", "transactiondate"] + cat_cols, axis=1)
feat_names = test_df.columns.values

for c in test_df.columns:
    if test_df[c].dtype == 'object':
        test_df[c] = LabelEncoder().fit_transform(list(test_df[c].values))

preds = {}
for name, model in modelling.stage1_models.items():
    print(name)
    with open("models/{}.py".format(name), "rb") as f:
        model = pkl.load(f)
    preds[name] = model.predict(test_df)

test_df = pd.DataFrame(preds)
preds = {}
for name, model in modelling.stage2_models.items():
    print(name)
    with open("models/{}.py".format(name), "rb") as f:
        model = pkl.load(f)
    preds[name] = model.predict(test_df).astype(np.float32)
preds = pd.DataFrame(preds).mean(axis=1)
preds["ParcelId"] = parcelid
preds["date"] = date

preds.to_csv("test_preds.csv")