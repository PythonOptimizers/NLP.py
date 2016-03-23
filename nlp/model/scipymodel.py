"""Models with sparse matrices in SciPy coordinate (COO) format."""

from scipy import sparse as sp

from nlp.model.nlpmodel import NLPModel
from nlp.model.amplpy import AmplModel
from nlp.model.snlp import SlackModel
from nlp.model.qnmodel import QuasiNewtonModel
from pykrylov.linop.linop import linop_from_ndarray
import numpy as np


class SciPyNLPModel(NLPModel):
    """`NLPModel` with sparse matrices in SciPy coordinate (COO) format.

    The `NLPModel`'s :meth:`jac` and :meth:`hess` methods
    should return that sparse Jacobian and Hessian in coordinate format:
    (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        """Evaluate Lagrangian Hessian."""
        vals, rows, cols = super(SciPyNLPModel, self).hess(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.nvar, self.nvar))

    def jac(self, *args, **kwargs):
        """Evaluate sparse constraints Jacobian."""
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))
        vals, rows, cols = super(SciPyNLPModel, self).jac(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))


class SciPyAmplModel(AmplModel):
    """`AmplModel` with sparse matrices n SciPy coordinate (COO) format.

    The `AmplModel`'s :meth:`jac` and :meth:`hess` methods
    should return that sparse Jacobian and Hessian in coordinate format:
    (vals, rows, cols).
    """

    # MRO: 1. SciPyAmplModel
    #      2. AmplModel
    #      3. NLPModel

    def A(self, *args, **kwargs):
        """Evaluate sparse Jacobian of the linear part of the constraints.

        Useful to obtain constraint matrix when problem is a linear programming
        problem.
        """
        vals, rows, cols = super(SciPyAmplModel. self).A(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def jac(self, *args, **kwargs):
        """Evaluate sparse constraints Jacobian."""
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))

        vals, rows, cols = super(SciPyAmplModel, self).jac(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def hess(self, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z).

        AMPL only returns lower triangular part of the Hessian and
        `scipy.coo_matrix` doesn't have a `symmetric` attributes, so we need to
        copy the upper part of the matrix
        """
        l_vals, l_rows, l_cols = super(SciPyAmplModel, self).hess(*args,
                                                                  **kwargs)

        indices = []
        for i in xrange(len(l_rows)):
            if l_rows[i] == l_cols[i]:
                indices.append(i)

        # stricly upper triangular part of H is obtained by switching rows and
        # cols indices and removing values on the diagonal.
        u_rows = np.delete(l_cols, indices)  # a copy is done
        u_cols = np.delete(l_rows, indices)
        u_vals = np.delete(l_vals, indices)

        H = sp.coo_matrix((np.concatenate((l_vals, u_vals)),
                          (np.concatenate((l_rows, u_rows)),
                           np.concatenate((l_cols, u_cols)))),
                          shape=(self.nvar, self.nvar))
        return H

    def jop(self, *args, **kwargs):
        """Obtain Jacobian at x as a linear operator."""
        return self.jac(*args, **kwargs)


class SciPySlackModel(SlackModel):
    """SlackModel in wich matrices are SciPy matrices.

    :keywords:
        :model:  Original model to be transformed into a slack form.

    """

    def __init__(self, model, **kwargs):
        if not isinstance(model, NLPModel):
            raise TypeError("The model in `model` should be a SciPyNLPModel"
                            "or a derived class of it.")
        super(SciPySlackModel, self).__init__(model)

    def _jac(self, x, lp=False):
        """Helper method to assemble the Jacobian matrix.

        See the documentation of :meth:`jac` for more information.
        The positional argument `lp` should be set to `True` only if the
        problem is known to be a linear program. In this case, the evaluation
        of the constraint matrix is cheaper and the argument `x` is ignored.

        """
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))

        model = self.model
        on = self.original_n

        # Get contribution of general constraints
        J = model.jac(x, lp)
        c_vals = J.data
        c_rows = J.row
        c_cols = J.col

        # Create a few index lists
        lowerC = np.array(model.lowerC, dtype=np.int64)
        nlowerC = model.nlowerC
        upperC = np.array(model.upperC, dtype=np.int64)
        nupperC = model.nupperC
        rangeC = np.array(model.rangeC, dtype=np.int64)
        nrangeC = model.nrangeC
        rlowerC = np.array(range(nlowerC))
        rupperC = np.array(range(nupperC))
        rrangeC = np.array(range(nrangeC))

        # Insert contribution of slacks on general constraints
        rows = np.concatenate((c_rows, lowerC, upperC, rangeC))
        cols = np.concatenate((c_cols, on + rlowerC, on + nlowerC + rupperC,
                               on + nlowerC + nupperC + rrangeC))
        vals = np.concatenate((c_vals,
                               -1.0*np.ones(nlowerC + nupperC + nrangeC)))

        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        if z is None:
            z = np.zeros(self.m)

        H = model.hess(x, z, **kwargs)
        vals = H.data
        rows = H.row
        cols = H.col
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.nvar, self.nvar))
