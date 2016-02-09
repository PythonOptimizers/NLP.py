from scipy import sparse as sp

from nlp.model.nlpmodel import NLPModel
from nlp.model.amplpy import AmplModel
from pykrylov.linop.linop import linop_from_ndarray
import numpy as np


class SciPyNLPModel(NLPModel):
    """
    An `NLPModel` where sparse matrices are returned in SciPy
    coordinate (COO) format. The `NLPModel`'s `jac` and `hess` methods
    should return that sparse Jacobian and Hessian in coordinate format:
    (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        vals, rows, cols = super(SciPyNLPModel, self).hess(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.nvar, self.nvar))

    def jac(self, *args, **kwargs):
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))
        vals, rows, cols = super(SciPyNLPModel, self).jac(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))


class SciPyAmplModel(AmplModel):
    # MRO: 1. SciPyAmplModel
    #      2. AmplModel
    #      3. NLPModel
    #

    def A(self, *args, **kwargs):
        """
        Evaluate sparse Jacobian of the linear part of the
        constraints. Useful to obtain constraint matrix
        when problem is a linear programming problem.
        """
        vals, rows, cols = super(SciPyAmplModel. self).A(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def jac(self, *args, **kwargs):
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))

        vals, rows, cols = super(SciPyAmplModel, self).jac(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def hess(self, *args, **kwargs):
        """
        AMPL only returns lower triangular part of the Hessian and `scipy.coo_matrix`
        doesn't have a `symmetric` attributes, so we need to copy the upper part of
        the matrix.
        """
        l_vals, l_rows, l_cols = super(SciPyAmplModel, self).hess(*args, **kwargs)

        indices = []
        for i in xrange(len(l_rows)):
            if l_rows[i] == l_cols[i]:
                indices.append(i)

        # stricly upper triangular part of H is obtained by switching rows and
        # cols indices and removing values on the diagonal.
        u_rows = np.delete(l_cols, indices)  # np.delete remove items (a copy is done)
        u_cols = np.delete(l_rows, indices)
        u_vals = np.delete(l_vals, indices)

        H = sp.coo_matrix((np.concatenate((l_vals, u_vals)),
                          (np.concatenate((l_rows, u_rows)),
                           np.concatenate((l_cols, u_cols)))),
                          shape=(self.nvar, self.nvar))
        return H

    def jop(self, *args, **kwargs):
        return self.jac(*args, **kwargs)
