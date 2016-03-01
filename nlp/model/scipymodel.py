"""Models where sparse matrices are returned in SciPy coordinate format."""


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
        """Evaluate sparse Jacobian of the linear part of the constraints.

        Useful to obtain constraint matrix when problem is a linear programming
        problem.

        """
        vals, rows, cols = super(SciPyAmplModel. self).A(*args, **kwargs)
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.ncon, self.nvar))

    def jac(self, *args, **kwargs):
        """Evaluate constraints Jacobian at x."""
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
        if not isinstance(model, SciPyNLPModel):
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
        n = self.n
        m = self.m
        model = self.model
        on = self.original_n

        lowerC = np.array(model.lowerC)
        nlowerC = model.nlowerC
        upperC = np.array(model.upperC)
        nupperC = model.nupperC
        rangeC = np.array(model.rangeC)
        nrangeC = model.nrangeC

        # Initialize sparse Jacobian
        if self.ncon == 0:  # SciPy cannot create sparse matrix of size 0.
            return linop_from_ndarray(np.empty((0, self.nvar), dtype=np.float))

        J = sp.coo_matrix((self.ncon, self.nvar))

        # Insert contribution of general constraints
        if lp:
            J[:on, :on] = self.model.A()
        else:
            J[:on, :on] = self.model.jac(x[:on])

        # Create a few index lists
        rlowerC = np.array(range(nlowerC))
        rupperC = np.array(range(nupperC))
        rrangeC = np.array(range(nrangeC))

        # Insert contribution of slacks on general constraints
        J[lowerC, on + rlowerC] = -1.0
        J[upperC, on + nlowerC + rupperC] = -1.0
        J[rangeC, on + nlowerC + nupperC + rrangeC] = -1.0

        return J

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        on = model.n

        pi = self.convert_multipliers(z)
        H = sp.coo_matrix((self.n, self.n))
        H[:on, :on] = self.model.hess(x[:on], pi, *args, **kwargs)
        return H
