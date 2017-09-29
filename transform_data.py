import numpy as np
import pandas as pd

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


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


def add_date_features(date_error, df):
    df["error_this_month"] = dd.merge(df[["transactiondate"]], date_error.rolling(30).mean().shift(),
                                      left_on="transactiondate", right_index=True, how="left")["logerror"]
    df["error_last_month"] = dd.merge(df[["transactiondate"]], date_error.rolling(30).mean().shift(31),
                                      left_on="transactiondate", right_index=True, how="left")["logerror"]
    return df

from dask import dataframe as dd

props = dd.read_csv(r"input/properties_2016.csv", blocksize=2**25, assume_missing=True)

train = pd.read_csv(r"input/train_2016_v2.csv")
train["transactiondate"] = pd.to_datetime(train["transactiondate"])
train = dd.from_pandas(train[["parcelid", "transactiondate", "logerror"]], npartitions=8)


dates = train.groupby("transactiondate")[["logerror"]].mean()

train = dd.merge(train, props, on='parcelid', how='left')
train = add_date_features(dates, train)
train = add_features(train)
# Average structuretaxvaluedollarcnt by city
#group = train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
#train['N-Avg-structuretaxvaluedollarcnt'] = train['regionidcity'].map(group)

# Deviation away from average
#train['N-Dev-structuretaxvaluedollarcnt'] = abs(
#    (train['structuretaxvaluedollarcnt'] - train['N-Avg-structuretaxvaluedollarcnt'])) / train[
#                                                   'N-Avg-structuretaxvaluedollarcnt']

train.to_csv("input/train.*.csv")
del train

from datetime import timedelta

samp = pd.read_csv(r"input/sample_submission.csv").melt(id_vars="ParcelId")
samp["transactiondate"] = pd.to_datetime(samp["variable"].apply(lambda x: "{}-{}-1".format(x[:4], int(x[4:])))) + timedelta(days=30)
samp = dd.from_pandas(samp[["ParcelId", "transactiondate"]], npartitions=16)
print(samp.tail())
print(dates.tail())
samp = dd.merge(samp, props, left_on="ParcelId", right_on='parcelid', how='left')
del props
samp = add_date_features(dates, samp)
samp = add_features(samp)
#samp['N-Avg-structuretaxvaluedollarcnt'] = samp['regionidcity'].map(group)

# Deviation away from average
#samp['N-Dev-structuretaxvaluedollarcnt'] = abs(
#    (samp['structuretaxvaluedollarcnt'] - samp['N-Avg-structuretaxvaluedollarcnt'])) / samp[
#                                                   'N-Avg-structuretaxvaluedollarcnt']

samp.to_csv("input/test.*.csv")
