try:
    from cysparse.sparse.ll_mat import LLSparseMatrix
    import cysparse.common_types.cysparse_types as types
except:
    print "CySparse is not installed!"

from nlp.model.nlpmodel import NLPModel
from nlp.model.snlp import SlackModel
from nlp.model.qnmodel import QuasiNewtonModel
from nlp.model.amplpy import AmplModel
from pykrylov.linop import CysparseLinearOperator
import numpy as np


class CySparseNLPModel(NLPModel):
    """
    An `NLPModel` where sparse matrices are returned as CySparse matrices.
    The `NLPModel`'s `jac` and `hess` methods should return sparse
    Jacobian and Hessian in coordinate format: (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z).

        Note that `rows`, `cols` and `vals` must represent a LOWER triangular
        sparse matrix in the coordinate format (COO).
        """
        vals, rows, cols = super(CySparseNLPModel, self).hess(*args, **kwargs)
        H = LLSparseMatrix(size=self.nvar, size_hint=vals.size,
                           store_symmetric=True, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)
        H.put_triplet(rows, cols, vals)
        return H

    def jac(self, *args, **kwargs):
        """Evaluate constraints Jacobian at x."""
        vals, rows, cols = super(CySparseNLPModel, self).jac(*args, **kwargs)
        J = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar,
                           size_hint=vals.size, store_symmetric=False,
                           itype=types.INT64_T, dtype=types.FLOAT64_T)
        J.put_triplet(rows, cols, vals)
        return J


class CySparseAmplModel(CySparseNLPModel, AmplModel):
    # MRO: 1. CySparseAmplModel
    #      2. CySparseNLPModel
    #      3. AmplModel
    #      4. NLPModel
    #
    # Here, `jac` and `hess` are inherited directly from CySparseNPLModel.
    #

    def A(self, *args, **kwargs):
        """
        Evaluate sparse Jacobian of the linear part of the
        constraints. Useful to obtain constraint matrix
        when problem is a linear programming problem.
        """
        vals, rows, cols = super(CySparseAmplModel. self).A(*args, **kwargs)
        A = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar,
                           size_hint=vals.size, store_symmetric=False,
                           type=types.INT64_T, dtype=types.FLOAT64_T)
        A.put_triplet(rows, cols, vals)
        return A

    def jop(self, *args, **kwargs):
        """Obtain Jacobian at x as a linear operator."""
        return CysparseLinearOperator(self.jac(*args, **kwargs))


class CySparseSlackModel(SlackModel):
    """
    Reformulate an optimization problem using slack variables.

    New model represents matrices as `CySparse` matrices.

    :parameters:
        :model:  Original model to be transformed into a slack form.

    """

    def __init__(self, model, **kwargs):

        if not isinstance(model, CySparseNLPModel):
            raise TypeError("The model in `model` should be a CySparseNLPModel"
                            "or a derived class of it.")
        super(CySparseSlackModel, self).__init__(model)

    def _jac(self, x, lp=False):
        """
        Helper method to assemble the Jacobian matrix.
        See the documentation of :meth:`jac` for more information.

        The positional argument `lp` should be set to `True` only if the
        problem is known to be a linear program. In this case, the evaluation
        of the constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.n
        m = self.m
        model = self.model
        on = self.original_n

        lowerC = np.array(model.lowerC, dtype=np.int64)
        nlowerC = model.nlowerC
        upperC = np.array(model.upperC, dtype=np.int64)
        nupperC = model.nupperC
        rangeC = np.array(model.rangeC, dtype=np.int64)
        nrangeC = model.nrangeC

        # Initialize sparse Jacobian
        nnzJ = self.model.nnzj + m
        J = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar, size_hint=nnzJ,
                           store_symmetric=False, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)

        # Insert contribution of general constraints
        if lp:
            J[:on, :on] = self.model.A()
        else:
            J[:on, :on] = self.model.jac(x[:on])

        # Create a few index lists
        rlowerC = np.array(range(nlowerC), dtype=np.int64)
        rupperC = np.array(range(nupperC), dtype=np.int64)
        rrangeC = np.array(range(nrangeC), dtype=np.int64)

        # Insert contribution of slacks on general constraints
        J.put_triplet(lowerC, on + rlowerC,
                      -1.0*np.ones(nlowerC, dtype=np.float64))
        J.put_triplet(upperC, on + nlowerC + rupperC,
                      -1.0*np.ones(nupperC, dtype=np.float64))
        J.put_triplet(rangeC, on + nlowerC + nupperC + rrangeC,
                      -1.0*np.ones(nrangeC, dtype=np.float64))

        return J

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        if z is None:
            z = np.zeros(self.m)

        on = model.n

        H = LLSparseMatrix(size=self.nvar, size_hint=self.model.nnzh,
                           store_symmetric=True, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)
        H[:on, :on] = self.model.hess(x[:on], z, *args, **kwargs)
        return H
