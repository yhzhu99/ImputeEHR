import h2o
from sklearn.decomposition import NMF

from impute_ehr.data import preprocess
from h2o.estimators import H2OGeneralizedLowRankEstimator


class NMFImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None):
        h2o.init()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True
        self.imputer = H2OGeneralizedLowRankEstimator(k=4,
                                                      loss="quadratic",
                                                      gamma_x=0.5,
                                                      gamma_y=0.5,
                                                      max_iterations=700,
                                                      recover_svd=True,
                                                      init="SVD",
                                                      transform="standardize")

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
        The fitted `NMFImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
        ds = h2o.H2OFrame(ds)
        self.imputer = self.imputer.train(training_frame=ds)
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
        ds = h2o.H2OFrame(ds)
        ds = self.imputer.transform_frame(ds)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds
