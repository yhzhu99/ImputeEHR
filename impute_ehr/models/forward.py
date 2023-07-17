import pandas as pd
import numpy as np
from impute_ehr.data import preprocess


class ForwardImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = False
        self.require_val = False
        self.require_save_model = False

    def fit(self):
        pass

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
        ds = pd.DataFrame(ds)
        ds = ds.fillna(method='ffill', axis=0)
        ds = ds.fillna(method='ffill', axis=1)
        ds = preprocess.reverse_flatten_to_matrix(ds.to_numpy(), lens)
        # print(lens)
        return ds
