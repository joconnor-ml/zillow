import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, MinMaxScaler, PolynomialFeatures, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
import logging


from sklearn.base import BaseEstimator, RegressorMixin, clone


class ReducedFeatures(BaseEstimator, RegressorMixin):
    """Fit separate models for rows with missing values in the selected columns.
    At test time, use predictions from the relevant model depending on whether
    value is missing or not."""
    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = features
        self.models = {}

    def fit(self, X, y, **kwargs):
        for feature in self.features:
            filt = X[feature].isnull()
            temp_X = X.drop(feature, axis=1)
            temp_y = y
            self.models[feature] = clone(self.estimator).fit(temp_X, temp_y, **kwargs)
            self.main_model = clone(self.estimator).fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = np.zeros(X.shape[0])
        for feature in self.features:
            filt = X[feature].isnull()
            temp_X = X[filt].drop(feature, axis=1)
            preds[filt] = self.models[feature].predict(temp_X, **kwargs)
            if (~filt).sum() > 0:
                preds[~filt] = self.main_model.predict(X[~filt], **kwargs)
        return preds


class GroupedModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, groupby):
        self.estimator = estimator
        self.groupby = groupby
        self.models = {}

    def fit(self, X, y, **kwargs):
        for name in X[self.groupby].unique():
            filt = X[self.groupby] == name
            self.models[name] = clone(self.estimator).fit(X[filt], y[filt], **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = np.zeros(X.shape[0])
        for name in X[self.groupby].unique():
            filt = X[self.groupby] == name
            preds[filt] = self.models[name].predict(X[filt], **kwargs)
        return preds

    def predict_proba(self, X, **kwargs):
        preds = np.zeros(X.shape[0])
        for name in X[self.groupby].unique():
            filt = X[self.groupby] == name
            preds[filt] = self.models[name].predict(X[filt], **kwargs)
        return preds


class NanGroupedModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator
        self.cluster = KMeans(n_clusters=2)
        self.models = {}

    def fit(self, X, y, **kwargs):
        nans = X.isnull()
        clusters = self.cluster.fit_transform(nans).argmin(axis=1)
        for name in np.unique(clusters):
            filt = clusters == name
            self.models[name] = clone(self.estimator).fit(X[filt], y[filt], **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = np.zeros(X.shape[0])
        nans = X.isnull()
        clusters = self.cluster.transform(nans).argmax(axis=1)
        for name in np.unique(clusters):
            filt = clusters == name
            preds[filt] = self.models[name].predict(X[filt], **kwargs)
        return preds

    def predict_proba(self, X, **kwargs):
        preds = np.zeros(X.shape[0])
        nans = X.isnull()
        clusters = self.cluster.transform(nans).argmax(axis=1)
        for name in np.unique(clusters):
            filt = clusters == name
            preds[filt] = self.models[name].predict(X[filt], **kwargs)
        return preds


class TargetShiftingModel(BaseEstimator, RegressorMixin):
    def init(self, estimator, mean, spread):
        self.estimator = estimator
        self.mean = mean
        self.spread = spread

    def fit(self, X, y, **kwargs):
        # shift distribution of y to match that of the expected target
        y_renorm = y.sub(y.mean()).mul(1/(y.quantile(0.75) - y.quantile(0.25)))
        y_renorm = y_renorm.mul(self.spread).add(self.mean)

        self.estimator = self.estimator.fit(X, y_renorm, **kwargs)

    def predict(self, X, **kwargs):
        return self.estimator.predict(X, **kwargs)


class MaskingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, min_quantile, max_quantile):
        self.estimator = estimator
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    def fit(self, X, y, **kwargs):
        # shift distribution of y to match that of the expected target
        filt = y.between(np.percentile(y, self.min_quantile*100),
                         np.percentile(y, self.max_quantile*100))
        self.estimator = self.estimator.fit(X.values[filt], y.values[filt], **kwargs)

    def predict(self, X, **kwargs):
        return self.estimator.predict(X.values, **kwargs)
