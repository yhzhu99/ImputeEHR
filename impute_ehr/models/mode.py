import numpy as np
from sklearn.impute import SimpleImputer
from impute_ehr.data import preprocess


class ModeImpute:
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
        model = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        ds, lens = preprocess.flatten_to_matrix(ds)
        model.fit(ds)
        ds = model.transform(ds)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds
