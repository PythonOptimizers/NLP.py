try:
    from pysparse.sparse import PysparseMatrix as psp
except:
    print "PySparse is not installed!"

from nlp.model.nlpmodel import NLPModel
from nlp.model.snlp import SlackModel
from nlp.model.qnmodel import QuasiNewtonModel
from nlp.model.amplpy import AmplModel
from pykrylov.linop.linop import PysparseLinearOperator

import numpy as np


class PySparseNLPModel(NLPModel):
    """
    An `NLPModel` where sparse matrices are returned in PySparse format.
    The `NLPModel`'s `jac` and `hess` methods should return that sparse
    Jacobian and Hessian in coordinate format: (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        vals, rows, cols = super(PySparseNLPModel, self).hess(*args, **kwargs)
        H = psp(size=self.nvar, sizeHint=vals.size, symmetric=True)
        H.put(vals, rows, cols)
        return H

    def jac(self, *args, **kwargs):
        vals, rows, cols = super(PySparseNLPModel,
                                 self).jac(*args, **kwargs)
        J = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        J.put(vals, rows, cols)
        return J


class PySparseAmplModel(PySparseNLPModel, AmplModel):
    # MRO: 1. PySparseAmplModel
    #      2. PySparseNLPModel
    #      3. AmplModel
    #      4. NLPModel
    #
    # Here, `jac` and `hess` are inherited directly from PySparseNPLModel.
    #

    def __init__(self, *args, **kwargs):
        super(PySparseAmplModel, self).__init__(*args, **kwargs)

    def A(self, *args, **kwargs):
        """
        Evaluate sparse Jacobian of the linear part of the
        constraints. Useful to obtain constraint matrix
        when problem is a linear programming problem.
        """
        vals, rows, cols = super(PySparseNLPModel,
                                 self).A(*args, **kwargs)
        A = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        A.put(vals, rows, cols)
        return A

    def jop(self, *args, **kwargs):
        return PysparseLinearOperator(self.jac(*args, **kwargs))


class PySparseSlackModel(SlackModel):
    def __init__(self, model, keep_variable_bounds=False, **kwargs):
        if not isinstance(model, PySparseNLPModel):
            raise TypeError("The model in `model` should be a PySparseNLPModel"
                            "or a derived class of it.")
        kvb = keep_variable_bounds
        super(PySparseSlackModel, self).__init__(model,
                                                 keep_variable_bounds=kvb)

    def _jac(self, x, lp=False):
        """
        Helper method to assemble the Jacobian matrix of the constraints of the
        transformed problems. See the documentation of :meth:`jac` for more
        information.

        The positional argument `lp` should be set to `True` only if the
        problem is known to be a linear program. In this case, the evaluation
        of the constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.n
        m = self.m
        model = self.model
        on = self.model.n
        om = self.model.m

        lowerC = np.array(model.lowerC)
        nlowerC = model.nlowerC
        upperC = np.array(model.upperC)
        nupperC = model.nupperC
        rangeC = np.array(model.rangeC)
        nrangeC = model.nrangeC
        lowerB = np.array(model.lowerB)
        nlowerB = model.nlowerB
        upperB = np.array(model.upperB)
        nupperB = model.nupperB
        rangeB = np.array(model.rangeB)
        nrangeB = model.nrangeB
        nbnds = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        # Overestimate of nnz elements in Jacobian
        nnzJ = 2 * self.model.nnzj + m + nrangeC + nbnds + nrangeB

        # Initialize sparse Jacobian
        J = psp(nrow=m, ncol=n, sizeHint=nnzJ)

        # Insert contribution of general constraints.
        if lp:
            J[:om, :on] = self.model.A()
        else:
            J[:om, :on] = self.model.jac(x[:on])
        J[upperC, :on] *= -1.0
        J[om:om + nrangeC, :on] = J[rangeC, :on]  # upper side of range const.
        J[om:om + nrangeC, :on] *= -1.0

        # Create a few index lists
        rlowerC = np.array(range(nlowerC))
        rlowerB = np.array(range(nlowerB))
        rupperC = np.array(range(nupperC))
        rupperB = np.array(range(nupperB))
        rrangeC = np.array(range(nrangeC))
        rrangeB = np.array(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(-1.0,      lowerC,  on + rlowerC)
        J.put(-1.0,      upperC,  on + nlowerC + rupperC)
        J.put(-1.0,      rangeC,  on + nlowerC + nupperC + rrangeC)
        J.put(-1.0, om + rrangeC, on + nlowerC + nupperC + nrangeC + rrangeC)

        if self.keep_variable_bounds:
            return J

        # Insert contribution of bound constraints on the original problem
        bot = om+nrangeC
        J.put(1.0, bot + rlowerB, lowerB)
        bot += nlowerB
        J.put(1.0, bot + rrangeB, rangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, upperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, rangeB)

        # Insert contribution of slacks on the bound constraints
        bot = om + nrangeC
        J.put(-1.0, bot + rlowerB, on + nSlacks + rlowerB)
        bot += nlowerB
        J.put(-1.0, bot + rrangeB, on + nSlacks + nlowerB + rrangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, on + nSlacks + nlowerB + nrangeB + rupperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, on+nSlacks+nlowerB+nrangeB+nupperB+rrangeB)
        return J

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate the Hessian of the Lagrangian."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        on = model.n

        pi = self.convert_multipliers(z)
        H = psp(nrow=self.n, ncol=self.n,
                symmetric=True,
                sizeHint=self.model.nnzh)
        H[:on, :on] = self.model.hess(x[:on], pi, *args, **kwargs)
        return H
