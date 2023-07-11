import pandas as pd
import h2o
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from h2o.estimators import H2OGeneralizedLowRankEstimator


class NMFImpute:
    def __init__(self, train_ds: pd.DataFrame):
        h2o.init()
        self.train_ds = h2o.H2OFrame(train_ds)
        self.require_fit = True
        self.imputer = H2OGeneralizedLowRankEstimator(k=4,
                                                      loss="quadratic",
                                                      gamma_x=0.5,
                                                      gamma_y=0.5,
                                                      max_iterations=700,
                                                      recover_svd=True,
                                                      init="SVD",
                                                      transform="standardize")

    def fit(self):
        ds = self.train_ds.ascharacter().asnumeric()
        # cols of time datatype should not be involved in NMF.
        ds = ds[:, 4:]
        self.imputer = self.imputer.train(training_frame=self.train_ds)

    def execute(self, ds: pd.DataFrame):
        ds = ds.copy(deep=True)
        ds = h2o.H2OFrame(ds)
        imputed_ds = ds[:, 4:]
        imputed_ds = self.imputer.transform_frame(imputed_ds)

        # get cols having datatype of time.
        # index should be reset to solve row mismatch bug
        rest_ds = ds[:, :4].as_data_frame()
        rest_ds.reset_index(drop=True, inplace=True)

        imputed_ds = imputed_ds.as_data_frame()

        return pd.concat([rest_ds, imputed_ds], axis=1)
