from sklearn.decomposition import NMF

from impute_ehr.data import preprocess

class NMFImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None, model=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True
        if model is None:
            self.imputer = NMF(n_components=2)
        else:
            self.imputer = model

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
        The fitted `NMFImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        self.imputer = self.imputer.fit(ds)
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
        ds = self.imputer.transform(ds)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds
