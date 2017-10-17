import dask
from dask import dataframe as dd
from zillow.data_utils import add_features, add_date_features

if __name__ == "__main__":
    import sys
    arg = sys.argv[1]
    if arg not in ["train", "test", "train_2017", "test_2017"]:
        sys.exit(1)

    with dask.set_options(get=dask.get):
        print(r"input/{}.*.hdf".format(arg))
        df = dd.read_hdf(r"input/{}.*.hdf".format(arg), "data", chunksize=1000000)#.set_index("ParcelId")
        df = add_features(df)
        df = add_date_features(df)
        print(df.head())
        df.to_hdf(r"input/{}2.*.hdf".format(arg), "data")
