import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from zillow import modelling
import pickle as pkl
import dask.dataframe as dd

test_df = dd.read_hdf("input/test_20172.*.hdf", "data")

with open("input/feat_names.pkl", "rb") as f:
    feat_names = pkl.load(f)

with open("input/feat_names_both.pkl", "rb") as f:
    feat_names_both = pkl.load(f)

test_df = test_df

with open("input/encoders.pkl", "rb") as f:
    encoders = pkl.load(f)

for col, encoder in encoders.items():
    print(encoder.classes_)
    print(test_df[col].head(1000, npartitions=10).unique())
    test_df[col] = test_df.map_partitions(lambda x: encoder.transform(list(x[col].replace(np.nan, encoder.classes_[0]).values)),
                                          meta=pd.Series(dtype=np.float32))

with open("input/log_cols.pkl", "rb") as f:
    log_cols = pkl.load(f)

for c in log_cols:
    test_df[c] = test_df.map_partitions(lambda x: np.log(x[c] + 1),
                                        meta=pd.Series(dtype=np.float32))

test_df = test_df.map_partitions(lambda x: x.replace(np.inf, np.nan).replace(-np.inf, np.nan))

preds = {}
for name, model in modelling.stage1_models.items():
    for prefix in [""]:
        if prefix == "both_":
            feats = list(feat_names_both)
        else:
            feats = list(feat_names)
        print(prefix+name)
        with open("models/{}.py".format(prefix+name), "rb") as f:
            model = pkl.load(f)
        try:
            preds[prefix+name] = test_df.map_partitions(lambda x: model.predict(x[feats]),
                                                        meta=pd.Series(dtype=np.float32)).compute()
        except Exception as e:
            print(e)

del test_df

test_df = pd.DataFrame(preds)
test_df.to_hdf("stack_stage1_test.hdf", "data")
