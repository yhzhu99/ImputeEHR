import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from impute_ehr.data import preprocess

class PCAImpute(BaseEstimator, TransformerMixin):
    def __init__(self, train_ds: list = None, val_ds: list = None, model=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True

        # this imputer is a list.
        # imputer[0] is a PCA model.
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
            The fitted `PCAImputer` class instance.
        """
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
            The fitted `PCAImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        X = check_array(ds, dtype=np.float64,
                        force_all_finite=False)

        X_nan = np.isnan(X)
        most_by_nan = X_nan.sum(axis=0).argsort()[::-1]

        imputed = self.imputer[1].fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []

        self.imputer[0] = PCA(n_components=int(
            np.sqrt(min(X.shape))), whiten=True)

        for iter in range(self.max_iter):
            # rebuild the matrix to update the missing value
            estimator_ = self.imputer[0]
            estimator_.fit(new_imputed)
            new_imputed[X_nan] = estimator_.inverse_transform(
                estimator_.transform(new_imputed))[X_nan]

            # determine whether the model is convergent
            gamma = ((new_imputed - imputed) ** 2 /
                     (1e-6 + new_imputed.var(axis=0))).sum() / (1e-6 + X_nan.sum())
            self.gamma_.append(gamma)
            if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
                break

        return self

    def execute(self, ds: pd.DataFrame):
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
        if self.imputer is None:
            raise AttributeError("RandomForestImputer is expected before execute.")

        X = check_array(imputed_ds, copy=True, dtype=np.float64,
                        force_all_finite=False)

        X_nan = np.isnan(X)
        imputed = self.imputer[1].transform(X)

        estimator_ = self.imputer[0]
        X[X_nan] = estimator_.inverse_transform(estimator_.transform(imputed))[X_nan]

        return X