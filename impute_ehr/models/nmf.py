from impute_ehr.data import preprocess
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class NMFImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True
        self.imputer = NMF(n_components=15, init='nndsvd',
                           random_state=0, max_iter=250, solver='cd', tol=1e-4)

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
        The fitted `NMFImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        ds = np.array(ds)
        ds = SimpleImputer(strategy='mean').fit_transform(ds)
        self.imputer.fit(ds)
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
        ds = np.array(ds)
        ds = SimpleImputer(strategy='mean').fit_transform(ds)
        W = self.imputer.transform(ds)
        H = self.imputer.components_
        ds = np.dot(W, H)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds
