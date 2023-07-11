import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from impute_ehr.data import preprocess


class RandomForestImpute(BaseEstimator, TransformerMixin):
    def __init__(self, train_ds: list = None, val_ds: list = None, model=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True

        # this imputer is a list.
        # imputer[0] is the RandomForestRegressor list, where every col of the ds has a RandomForestRegressor.
        # imputer[1] is just a SimpleImputer to impute the starting value before iteration.
        if model is None:
            self.imputer = [None, SimpleImputer(strategy='mean')]
        else:
            self.imputer = model

        self.max_iter = 10
        self.tol = 1e-3

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
            The fitted `RandomForestImpute` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        # make sure datatype of a col is float, na or inf
        X = check_array(ds, dtype=np.float64,
                        force_all_finite=False)

        # get the col number in descending order by the number of na.
        # [start:end:step], end=-1 means the last element, step=-1 means from back to front.
        X_nan = np.isnan(X)
        most_by_nan = X_nan.sum(axis=0).argsort()[::-1]

        # just impute X with SimpleImputer as the starting value
        imputed = self.imputer[1].fit_transform(X)
        new_imputed = imputed.copy()

        # get data of masked_array
        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []

        # RandomForestRegressor for each col
        self.imputer[0] = [RandomForestRegressor(
            n_estimators=20, n_jobs=-1, random_state=i) for i in range(X.shape[1])]

        # start iteration
        for iter in range(self.max_iter):
            # update estimator of each col
            for i in most_by_nan:
                # del the current col
                X_s = np.delete(new_imputed, i, 1)
                # nan rows in this col
                y_nan = X_nan[:, i]
                # data of ~nan rows
                X_train = X_s[~y_nan]
                y_train = new_imputed[~y_nan, i]
                # data of nan rows
                X_unk = X_s[y_nan]

                # train estimator[i]
                # use the imputed data from the last iteration as Y to carry out supervised learning
                estimator_ = self.imputer[0][i]
                estimator_.fit(X_train, y_train)
                # impute the nan row in this col with the corresponding estimator
                if len(X_unk) > 0:
                    new_imputed[y_nan, i] = estimator_.predict(X_unk)

            # determine whether the model is convergent
            gamma = ((new_imputed-imputed)**2 /
                     (1e-6+new_imputed.var(axis=0))).sum()/(1e-6+X_nan.sum())
            self.gamma_.append(gamma)
            if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
                break

        return self

    def execute(self, ds: list):
        """ Impute all missing values in ds.

        Parameters
        ----------
        ds : a nested list. [patient, visit, feature]

        Returns
        -------
        imputed ds : a nested list. [patient, visit, feature]
        """
        ds, lens = preprocess.flatten_to_matrix(ds)
        ds = self.transform(ds)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds

    def transform(self, imputed_ds: np.ndarray):
        """ The real execute function called by self.execute(). Impute dataset with valid col datatype.

        Parameters
        ----------
        imputed_ds : array-like of shape (n_samples, n_features)
            The input data to complete, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            All cols need to be imputed.

        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
        """
        # this class instance can work when it is fitted or inited with a given imputer, so we only need to check whether it has an imputer.
        # check_is_fitted(self, ['statistics_', 'imputer', 'gamma_'])
        if self.imputer is None:
            raise AttributeError("RandomForestImputer is expected before execute.")

        X = check_array(imputed_ds, copy=True, dtype=np.float64,
                        force_all_finite=False)
        # if X.shape[1] != self.statistics_.shape[1]:
        #     raise ValueError("X has %d features per sample, expected %d"
        #                      % (X.shape[1], self.statistics_.shape[1]))

        X_nan = np.isnan(X)
        imputed = self.imputer[1].transform(X)

        for i, estimator_ in enumerate(self.imputer[0]):
            X_s = np.delete(imputed, i, 1)
            y_nan = X_nan[:, i]

            X_unk = X_s[y_nan]
            if len(X_unk) > 0:
                X[y_nan, i] = estimator_.predict(X_unk)
        return X
