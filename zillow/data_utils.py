import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import dask
from dask import dataframe as dd


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    for col in props.columns:
        if props[col].dtype == 'object':
            props[col] = LabelEncoder().fit_transform(list(props[col].values)).astype(np.float32)

        if props[col].dtype != object and col != "parcelid":  # Exclude strings
            props[col] = props[col].astype(np.float32)

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


def get_properties(year=2016):
    if year == 2017:
        try:
            props = pd.read_hdf(r"input/processed_properties_2017.hdf", "data").set_index("parcelid", drop=True)
        except Exception as e:
            print(e)
            props = pd.read_csv(r"input/properties_2017.csv")  # , blocksize=2**25, assume_missing=True
            props = props.drop(props.columns[[22,32,34,49,55]], axis=1)#.set_index("parcelid", drop=True)
            props = reduce_mem_usage(props)
            props.to_hdf(r"input/processed_properties_2017.hdf", "data")
    else:
        try:
            props = pd.read_hdf(r"input/processed_properties_2016.hdf", "data")
        except Exception as e:
            print(e)
            props = pd.read_csv(r"input/properties_2016.csv")  # , blocksize=2**25, assume_missing=True
            props = props.drop(props.columns[[22,32,34,49,55]], axis=1).set_index("parcelid", drop=True)
            props = reduce_mem_usage(props)
            props.to_hdf(r"input/processed_properties_2016.hdf", "data")
    print(props.head())
    return props


def add_features(df_train):
    # life of property
    df_train['N-life'] = 2018 - df_train['yearbuilt']

    # error in calculation of the finished living area of home
    df_train['N-LivingAreaError'] = df_train['calculatedfinishedsquarefeet'] / df_train['finishedsquarefeet12']

    # proportion of living area
    df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet'] / df_train['lotsizesquarefeet']
    df_train['N-LivingAreaProp2'] = df_train['finishedsquarefeet12'] / df_train['finishedsquarefeet15']

    # Amout of extra space
    df_train['N-ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet']
    df_train['N-ExtraSpace-2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12']

    # Total number of rooms
    df_train['N-TotalRooms'] = df_train['bathroomcnt'] * df_train['bedroomcnt']

    # Average room size
    df_train['N-AvRoomSize'] = df_train['calculatedfinishedsquarefeet'] / df_train['roomcnt']

    # Number of Extra rooms
    df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms']

    # Ratio of the built structure value to land area
    df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt'] / df_train['landtaxvaluedollarcnt']

    # Does property have a garage, pool or hot tub and AC?
    df_train['N-GarPoolAC'] = ((df_train['garagecarcnt'] > 0) & (df_train['pooltypeid10'] > 0) & (
    df_train['airconditioningtypeid'] != 5)) * 1

    df_train["N-location"] = df_train["latitude"] + df_train["longitude"]
    df_train["N-location-2"] = df_train["latitude"] * df_train["longitude"]
    df_train["N-location-2round"] = df_train["N-location-2"].round(-4)

    df_train["N-latitude-round"] = df_train["latitude"].round(-4)
    df_train["N-longitude-round"] = df_train["longitude"].round(-4)

    # Ratio of tax of property over parcel
    df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt'] / df_train['taxamount']

    # TotalTaxScore
    df_train['N-TaxScore'] = df_train['taxvaluedollarcnt'] * df_train['taxamount']

    # polnomials of tax delinquency year
    df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2
    df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3

    # Length of time since unpaid taxes
    df_train['N-life'] = 2018 - df_train['taxdelinquencyyear']

    # Number of properties in the zip
    #zip_count = df_train['regionidzip'].value_counts().to_dict()
    #df_train['N-zip_count'] = df_train['regionidzip'].map(zip_count)

    # Number of properties in the city
    #city_count = df_train['regionidcity'].value_counts().to_dict()
    #df_train['N-city_count'] = df_train['regionidcity'].map(city_count)

    # Number of properties in the city
    #region_count = df_train['regionidcounty'].value_counts().to_dict()
    #df_train['N-county_count'] = df_train['regionidcounty'].map(city_count)

    # Indicator whether it has AC or not
    df_train['N-ACInd'] = (df_train['airconditioningtypeid'] != 5) * 1

    # Indicator whether it has Heating or not
    df_train['N-HeatInd'] = (df_train['heatingorsystemtypeid'] != 13) * 1

    # There's 25 different property uses - let's compress them down to 4 categories
    df_train['N-PropType'] = df_train.propertylandusetypeid.map(
        {31: "Mixed", 46: "Other", 47: "Mixed", 246: "Mixed", 247: "Mixed", 248: "Mixed", 260: "Home", 261: "Home",
         262: "Home", 263: "Home", 264: "Home", 265: "Home", 266: "Home", 267: "Home", 268: "Home", 269: "Not Built",
         270: "Home", 271: "Home", 273: "Home", 274: "Other", 275: "Home", 276: "Home", 279: "Home", 290: "Not Built",
         291: "Not Built"})

    # polnomials of the variable
    df_train["N-structuretaxvaluedollarcnt-2"] = df_train["structuretaxvaluedollarcnt"] ** 2
    df_train["N-structuretaxvaluedollarcnt-3"] = df_train["structuretaxvaluedollarcnt"] ** 3

    return df_train

def add_date_features(df):
    df["year"] = df["transactiondate"].str.slice(0,4).astype(np.int32)
    df["month"] = df["transactiondate"].str.slice(5,7).astype(np.int32)
    df["yearmonth"] = (df["year"] - 2016) * 12 + df["month"]

    date_error, date_high, date_low = get_dates()

    df["error_last_month"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-1, date_error,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["error_2ndlast_month"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-2, date_error,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["error_3rdlast_month"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-3, date_error,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["error_last_3months"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-1, date_error.rolling(3).mean(),
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["error_last_year"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-12, date_error,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["high_last_year"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-12, date_high,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    df["low_last_year"] = df.map_partitions(
        lambda x: pd.merge(x[["yearmonth"]]-12, date_low,
                           left_on="yearmonth", right_index=True, how="left")["logerror"]
    )
    return df


def get_dates():
    dates1 = pd.read_csv(r"input/train_2016_v2.csv")
    dates1["logerror"] += np.random.uniform(-0.0005, 0.0005, dates1.shape[0])

    dates1["year"] = dates1["transactiondate"].str.slice(0, 4).astype(np.int32)
    dates1["month"] = dates1["transactiondate"].str.slice(5, 7).astype(np.int32)
    dates1["yearmonth"] = (dates1["year"] - 2016)*12 + dates1["month"]
    dates1 = dates1.groupby("yearmonth")[["logerror"]].median()
    dateshigh1 = dates1.groupby("yearmonth")[["logerror"]].quantile(0.75)
    dateslow1 = dates1.groupby("yearmonth")[["logerror"]].quantile(0.25)

    dates2 = pd.read_csv(r"input/train_2017.csv")
    dates2["year"] = dates2["transactiondate"].str.slice(0, 4).astype(np.int32)
    dates2["month"] = dates2["transactiondate"].str.slice(5, 7).astype(np.int32)
    dates2["yearmonth"] = (dates2["year"] - 2016)*12 + dates2["month"]
    dates2 = dates2.groupby("yearmonth")[["logerror"]].median()
    dateshigh2 = dates2.groupby("yearmonth")[["logerror"]].quantile(0.75)
    dateslow2 = dates2.groupby("yearmonth")[["logerror"]].quantile(0.25)

    dates = pd.concat([dates1, dates2])
    dateshigh = pd.concat([dateshigh1, dateshigh2])
    dateslow = pd.concat([dateslow1, dateslow2])
    dates.to_csv("input/dates.csv")
    dateshigh.to_csv("input/dateshigh.csv")
    dateslow.to_csv("input/dateslow.csv")
    return dates.reset_index(), dateshigh.reset_index(), dateslow.reset_index()
