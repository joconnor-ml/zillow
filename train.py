import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from xgboost import XGBRegressor


train_df = pd.read_hdf("input/train.hdf", "data")

month = pd.to_datetime(train_df["transactiondate"]).dt.month
day = pd.to_datetime(train_df["transactiondate"]).dt.dayofyear

train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
feat_names = train_df.columns.values

for c in train_df.columns:
    if train_df[c].dtype == 'object':
        train_df[c] = LabelEncoder().fit_transform(list(train_df[c].values))

tolerance = 0.1
y = np.clip(train_y, np.median(train_y)-tolerance, np.median(train_y)+tolerance)

model.fit(train_df, y)

import pickle as pkl
with open("models/xgb.py", "wb") as f:
    pkl.dump(model, f)
