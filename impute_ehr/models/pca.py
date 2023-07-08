import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PCAImpute(BaseEstimator, TransformerMixin):
    def __init__(self, train_ds: pd.DataFrame):
        self.train_ds = train_ds
        self.initial_imputer = SimpleImputer(strategy='mean')
        self.require_fit = True
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
        ds = self.train_ds.copy(deep=True)
        # cols of time datatype should not be involved in PCA.
        ds = ds.iloc[:, 4:]
        X = check_array(ds, dtype=np.float64,
                        force_all_finite=False)

        X_nan = np.isnan(X)
        most_by_nan = X_nan.sum(axis=0).argsort()[::-1]

        imputed = self.initial_imputer.fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []

        self.estimators_ = [PCA(n_components=int(
            np.sqrt(min(X.shape))), whiten=True)]

        for iter in range(self.max_iter):
            if len(self.estimators_) > 1:
                for i in most_by_nan:

                    X_s = np.delete(new_imputed, i, 1)
                    y_nan = X_nan[:, i]

                    X_train = X_s[~y_nan]
                    y_train = new_imputed[~y_nan, i]
                    X_unk = X_s[y_nan]

                    estimator_ = self.estimators_[i]
                    estimator_.fit(X_train, y_train)
                    if len(X_unk) > 0:
                        new_imputed[y_nan, i] = estimator_.predict(X_unk)

            else:
                estimator_ = self.estimators_[0]
                estimator_.fit(new_imputed)
                new_imputed[X_nan] = estimator_.inverse_transform(
                    estimator_.transform(new_imputed))[X_nan]

            gamma = ((new_imputed-imputed)**2 /
                     (1e-6+new_imputed.var(axis=0))).sum()/(1e-6+X_nan.sum())
            self.gamma_.append(gamma)
            if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
                break

        return self

    def execute(self, ds: pd.DataFrame):
        ds = ds.copy(deep=True)

        # cols of time datatype should not be involved in PCA.
        imputed_ds = ds.iloc[:, 4:]

        # call the real execute function to impute ds
        X = self.transform(imputed_ds)

        # get imputed result with column name
        imputed_ds = pd.DataFrame(
            data=X,
            columns=imputed_ds.columns
        )
        # value should not be negative
        imputed_ds.where(imputed_ds >= 0, 0, inplace=True)

        # get cols having datatype of time.
        # index should be reset to solve row mismatch bug
        rest_ds = ds.iloc[:, :4]
        rest_ds.reset_index(drop=True, inplace=True)

        return pd.concat([rest_ds, imputed_ds], axis=1)

    # the real execute function. impute dataset with valid col datatype
    def transform(self, imputed_ds: pd.DataFrame):
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(imputed_ds, copy=True, dtype=np.float64,
                        force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[1]))

        X_nan = np.isnan(X)
        imputed = self.initial_imputer.transform(X)

        if len(self.estimators_) > 1:
            for i, estimator_ in enumerate(self.estimators_):
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]
                if len(X_unk) > 0:
                    X[y_nan, i] = estimator_.predict(X_unk)

        else:
            estimator_ = self.estimators_[0]
            X[X_nan] = estimator_.inverse_transform(
                estimator_.transform(imputed))[X_nan]

        return X
