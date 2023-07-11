import numpy as np

from impute_ehr.data import preprocess

class SoftImpute:
    def __init__(self, train_ds: list = None, val_ds: list = None, model=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.require_fit = True
        self.require_val = False
        self.require_save_model = True
        if model is None:
            self.imputer = SoftImputeSolver()
        else:
            self.imputer = model

    def fit(self):
        """ Fit the imputer on train_ds.

        Returns
        -------
        self : object
            The fitted `MICEImputer` class instance.
        """
        ds, lens = preprocess.flatten_to_matrix(self.train_ds)
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
        ds = self.imputer.transform(ds)
        ds = preprocess.reverse_flatten_to_matrix(ds, lens)
        return ds

# Soft Impute implementation
# Source: https://github.com/travisbrady/py-soft-impute/tree/master
#
# The code below is a partial implementation of the Soft Impute algorithm
# obtained from the GitHub repository mentioned above. Soft Impute is a matrix
# completion algorithm used for estimating missing values in a matrix by
# exploiting low-rank structures. It utilizes convex optimization techniques
# to find a low-rank approximation of the input matrix.
#
# The specific code snippet used here performs the Soft Thresholding step in
# the Soft Impute algorithm, which is responsible for applying a shrinkage
# operation to the singular values of the input matrix.
#
# Please make sure to consult the original repository for the complete code,
# licensing information, and any additional details regarding the Soft Impute
# algorithm.

def frob(Uold, Dsqold, Vold, U, Dsq, V):
    denom = (Dsqold ** 2).sum()
    utu = Dsq * (U.T.dot(Uold))
    vtv = Dsqold * (Vold.T.dot(V))
    uvprod = utu.dot(vtv).diagonal().sum()
    num = denom + (Dsqold ** 2).sum() - 2*uvprod
    return num / max(denom, 1e-9)

class SoftImputeSolver:
    def __init__(self, J=2, thresh=1e-05, lambda_=0, maxit=100, random_state=None, verbose=False):
        self.J = J
        self.thresh = thresh
        self.lambda_ = lambda_
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose
        self.u = None
        self.d = None
        self.v = None

    def fit(self, X):
        n,m = X.shape
        xnas = np.isnan(X)
        nz = m*n - xnas.sum()
        xfill = X.copy()
        V = np.zeros((m, self.J))
        U = self.rs.normal(0.0, 1.0, (n, self.J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        Dsq = np.ones((self.J, 1))
        col_means = np.nanmean(xfill, axis=0)
        np.copyto(xfill, col_means, where=np.isnan(xfill))
        ratio = 1.0
        iters = 0
        while ratio > self.thresh and iters < self.maxit:
            iters += 1
            U_old = U
            V_old = V
            Dsq_old = Dsq
            B = U.T.dot(xfill)

            if self.lambda_ > 0:
                tmp = (Dsq / (Dsq + self.lambda_))
                B = B * tmp

            Bsvd = np.linalg.svd(B.T, full_matrices=False)
            V = Bsvd[0]
            Dsq = Bsvd[1][:, np.newaxis]
            U = U.dot(Bsvd[2])

            tmp = Dsq * V.T

            xhat = U.dot(tmp)

            xfill[xnas] = xhat[xnas]
            A = xfill.dot(V).T
            Asvd = np.linalg.svd(A.T, full_matrices=False)
            U = Asvd[0]
            Dsq = Asvd[1][:, np.newaxis]
            V = V.dot(Asvd[2])
            tmp = Dsq * V.T

            xhat = U.dot(tmp)
            xfill[xnas] = xhat[xnas]
            ratio = frob(U_old, Dsq_old, V_old, U, Dsq, V)
            if self.verbose:
                print('iter: %4d ratio = %.5f' % (iters, ratio))

        self.u = U[:,:self.J]
        self.d = Dsq[:self.J]
        self.v = V[:,:self.J]
        return self

    def suv(self, vd):
        res = self.u.dot(vd.T)
        return res

    def transform(self, X, copyto=False):
        vd = self.v * np.outer(np.ones(self.v.shape[0]), self.d)
        X_imp = self.suv(vd)
        if copyto:
            np.copyto(X, X_imp, where=np.isnan(X))
        else:
            return X_imp
