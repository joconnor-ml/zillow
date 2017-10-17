import pandas as pd
import dask
from dask import dataframe as dd
from zillow.data_utils import get_properties


def reformat_date(x):
    year = int(x[:4])
    month = int(x[4:])
    return "{}-{}-01".format(year, month)

if __name__ == "__main__":
    props = get_properties()

    cols = ["ParcelId", "201610", "201611", "201612"]
    with dask.set_options(get=dask.get):
        samp = dd.read_csv(r"input/sample_submission.csv", usecols=cols, blocksize=1e6).set_index("ParcelId", drop=False)
        date_cols = samp.columns.drop("ParcelId")
        samp = dd.merge(samp, props, left_index=True, right_index=True, how='left')
        samp = samp.map_partitions(lambda x: x.melt(id_vars=list(props.columns) + ["ParcelId"], value_vars=date_cols))
        samp["transactiondate"] = samp["variable"].apply(reformat_date, meta=pd.Series(dtype=str))
        print(samp.tail())
        samp.drop(["variable", "value"], axis=1).to_hdf("input/test.*.hdf", "data")
