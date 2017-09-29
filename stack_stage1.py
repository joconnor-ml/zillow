import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from zillow import modelling
import pickle as pkl


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

cv = LeaveOneGroupOut()

preds = {}
for name, model in modelling.stage1_models.items():
    preds[name] = cross_val_predict(model, train_df, y, cv=cv, groups=month)
    print(name, np.abs((train_y - preds[name])).mean())
    model.fit(train_df, y)
    with open("models/{}.py".format(name), "wb") as f:
        pkl.dump(model, f)

del train_df


train_df = pd.DataFrame(preds)
train_df.to_csv("stack_stage1.csv")

preds = {}
for name, model in modelling.stage2_models.items():
    preds[name] = cross_val_predict(model, train_df, y, cv=cv, groups=month)
    print(name, np.abs((train_y - preds[name])).mean())
    model.fit(train_df, y)
    with open("models/{}.py".format(name), "wb") as f:
        pkl.dump(model, f)
