import pandas as pd
import dask
from dask import dataframe as dd
from zillow.data_utils import get_properties


if __name__ == "__main__":
    props = get_properties(2017)

    with dask.set_options(get=dask.get):
        train = dd.read_csv(r"input/train_2017.csv", blocksize=1e6).set_index("parcelid", drop=True)
        train = train.map_partitions(lambda x: pd.merge(x, props, left_index=True, right_index=True, how='left'))
        train.to_hdf("input/train_2017.*.hdf", "data")
