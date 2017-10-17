import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from zillow import modelling
import pickle as pkl
import dask.dataframe as dd

train_df = pd.concat([
    #dd.read_hdf("input/train2.*.hdf", "data").compute(),
    dd.read_hdf("input/train_20172.*.hdf", "data").compute(),
])
month = train_df["yearmonth"]

train_y = train_df['logerror']

tolerance = 100
y = np.clip(train_y,
            np.median(train_y)-tolerance,
            np.median(train_y)+tolerance)

cv = LeaveOneGroupOut()

train_df = pd.concat([
    pd.read_csv("stack_stage1_{}.csv".format(i), index_col=0)
    for i in [1,2,3]
], axis=1)

last_month = month.max()
filt = (month == last_month)

preds = {}
for name, model in modelling.stage2_models.items():
    preds[name] = cross_val_predict(model, train_df, y, cv=cv, groups=month//3)
    print(name, np.abs((train_y - preds[name])).mean(), np.abs((train_y - np.clip(preds[name], 0.006-0.1, 0.006+0.1))).mean())
    print("Last month:", name, np.abs((train_y[filt] - preds[name][filt])).mean())
    model.fit(train_df, y)
    if name == "lr2":
        imps = pd.Series(model.coef_, index=train_df.columns)
        print(imps)
    with open("models/{}.py".format(name), "wb") as f:
        pkl.dump(model, f)
    print()

preds = pd.DataFrame(preds)
print(np.abs((train_y - preds.mean(axis=1))).mean())