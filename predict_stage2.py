import pandas as pd
from zillow import modelling
import pickle as pkl
import dask.dataframe as dd

test_df = dd.read_hdf("input/test2.*.hdf", "data")
parcelid = test_df["ParcelId"].compute()
yearmonth = test_df["yearmonth"].compute()

test_df = pd.read_hdf("stack_stage1_test.hdf", "data")

preds = {}
for name, model in modelling.stage2_models.items():
    with open("models/{}.py".format(name), "rb") as f:
        model = pkl.load(f)
    print(name)
    preds[name] = model.predict(test_df)

del test_df

preds = pd.DataFrame(preds, index=parcelid.index)
preds["ParcelId"] = parcelid
preds["yearmonth"] = yearmonth
print(preds.max())
print(preds.min())
preds.to_hdf("preds_raw.hdf", "data")
