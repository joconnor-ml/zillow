from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor

stage1_models = {
    "reg": make_pipeline(Imputer(), StandardScaler(), Ridge()),
    "dummy": make_pipeline(Imputer(), DummyRegressor(strategy="constant", constant=0.006)),
    "xgb": XGBRegressor(max_depth=4, learning_rate=0.10, n_estimators=64),
    "lgb": LGBMRegressor(n_estimators=100, learning_rate=0.01,
                         boosting_type="gbdt", objective="regression",
                         metric="mae", sub_feature=0.5, num_leaves=60, min_data=500, min_hessian=1),
    #"lgb": LGBMRegressor(n_estimators=500, learning_rate=0.002,
    #                     boosting_type="gbdt", objective="regression",
    #                     metric="mae", sub_feature=0.5, num_leaves=60, min_data=500, min_hessian=1),
}


stage2_models = {
    "lr": LinearRegression(),
    "ridge": Ridge(),
}
