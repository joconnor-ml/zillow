from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge, LassoCV, SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from .model_utils import NanGroupedModel, MaskingModel

def dropnan(df):
    x = df.copy()
    try:
        x.loc[x.columns[x.isnull().sum() > 1000]] = 0
    except:
        pass
    return x


stage1_models = {
    #"lasso": make_pipeline(Imputer(), StandardScaler(), LassoCV(max_iter=10000)),
    #"lr": make_pipeline(Imputer(), StandardScaler(), LinearRegression()),
    #"dummy": make_pipeline(Imputer(), DummyRegressor(strategy="quantile", quantile=0.5)),
    "xgb": MaskingModel(XGBRegressor(max_depth=4, learning_rate=0.4, n_estimators=32, base_score=0.01),
                                     min_quantile=0.1, max_quantile=0.9),
    #"xgb_deep": MaskingModel(XGBRegressor(max_depth=6, learning_rate=0.04, n_estimators=128, base_score=0.01),
    #                                 min_quantile=0.1, max_quantile=0.9),
    #"xgb_deep": XGBRegressor(max_depth=6, learning_rate=0.04, n_estimators=128, base_score=0.006),
    #"gbq": make_pipeline(Imputer(),
    #                     GradientBoostingRegressor(alpha=0.5,
    #                                               max_depth=3, learning_rate=0.20, n_estimators=64,
    #                                               loss="quantile")),
    #"gbq_deep": make_pipeline(Imputer(),
    #                          GradientBoostingRegressor(alpha=0.5,
    #                                                    max_depth=6, learning_rate=0.10, n_estimators=64,
    #                                                    loss="quantile")),
    #"gbh": make_pipeline(Imputer(), GradientBoostingRegressor(alpha=0.5,
    #                                                          max_depth=8, learning_rate=0.05, n_estimators=100,
    #                                                          loss="huber")),
    #"lgb": MaskingModel(LGBMRegressor(n_estimators=256, learning_rate=0.005,
    #                     boosting_type="gbdt", objective="regression",
    #                     metric="mae", sub_feature=0.5, num_leaves=500, min_data=500, min_hessian=0.05,
    #                     bagging_fraction=0.85, bagging_freq=40),
    #                    min_quantile=0.1, max_quantile=0.9),
    ##"xgb_dropnan": NanGroupedModel(XGBRegressor(max_depth=8, learning_rate=0.10, n_estimators=64)),
    #"lgb_deep": LGBMRegressor(n_estimators=500, learning_rate=0.002,
    #                          boosting_type="gbdt", objective="regression",
    #                          metric="mae", sub_feature=0.5, num_leaves=60, min_data=500, min_hessian=1),
}


stage2_models = {
    #"lasso2": MaskingModel(make_pipeline(Imputer(), StandardScaler(), LassoCV()),
    #                       min_quantile=0.1, max_quantile=0.9),
    #"lr2": LinearRegression(fit_intercept=False),
    #"xgb2": MaskingModel(XGBRegressor(max_depth=4, learning_rate=0.10, n_estimators=64),
    #                     min_quantile=0.1, max_quantile=0.9),
    "gbq2": make_pipeline(Imputer(), GradientBoostingRegressor(alpha=0.5,
                                                               max_depth=3, learning_rate=0.10, n_estimators=64,
                                                               loss="quantile")),
}
