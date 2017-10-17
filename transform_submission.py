import pandas as pd
import sys

df = pd.read_hdf("preds_raw_2017.hdf", "data")
df["gbq2"] = pd.read_hdf("stack_stage1_test.hdf", "data")["xgb"]
df = df.pivot(index="ParcelId", columns="yearmonth", values="gbq2")
print(df.head())
df.columns = ["201710", "201711", "201712"]
for col in ["201610", "201611", "201612"]:
    df[col] = 0
df.loc[:, ["201610", "201611", "201612"]] = df[["201710", "201711", "201712"]].values
print(df.max())
print(df.min())

df.round(4).to_csv("final.csv.gz", compression="gzip")
df.add(0.001).round(4).to_csv("final_plus.csv.gz", compression="gzip")
