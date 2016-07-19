"""Models where sparse matrices are returned in PySparse format."""


try:
    from pysparse.sparse import PysparseMatrix as psp
except ImportError:
    print "PySparse is not installed!"

from nlp.model.nlpmodel import NLPModel
from nlp.model.snlp import SlackModel
from nlp.model.qnmodel import QuasiNewtonModel
from pykrylov.linop.linop import PysparseLinearOperator

import numpy as np


class PySparseNLPModel(NLPModel):
    """An `NLPModel` where sparse matrices are returned in PySparse format.

    The `NLPModel`'s `jac` and `hess` methods should return that sparse
    Jacobian and Hessian in coordinate format: (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        vals, rows, cols = super(PySparseNLPModel, self).hess(*args, **kwargs)
        H = psp(size=self.nvar, sizeHint=vals.size, symmetric=True)
        H.put(vals, rows, cols)
        return H

    def jac(self, *args, **kwargs):
        """Evaluate constraints Jacobian at x."""
        vals, rows, cols = super(PySparseNLPModel,
                                 self).jac(*args, **kwargs)
        J = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        J.put(vals, rows, cols)
        return J


try:
    from nlp.model.amplmodel import AmplModel

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
            """Evaluate sparse Jacobian of the linear part of the constraints.

            Useful to obtain constraint matrix when problem is a linear programming
            problem.
            """
            vals, rows, cols = super(PySparseAmplModel,
                                     self).A(*args, **kwargs)
            A = psp(nrow=self.ncon, ncol=self.nvar,
                    sizeHint=vals.size, symmetric=False)
            A.put(vals, rows, cols)
            return A

        def jop(self, *args, **kwargs):
            """Obtain Jacobian at x as a linear operator."""
            return PysparseLinearOperator(self.jac(*args, **kwargs))

    class QnPySparseAmplModel(QuasiNewtonModel, PySparseAmplModel):
        pass

except:
    pass


class PySparseSlackModel(SlackModel):
    """SlackModel in wich matrices are PySparse matrices.

    :keywords:
        :model:  Original model to be transformed into a slack form.

    """

    def __init__(self, model, **kwargs):
        if not isinstance(model, PySparseNLPModel):
            raise TypeError("The model in `model` should be a PySparseNLPModel"
                            "or a derived class of it.")
        super(PySparseSlackModel, self).__init__(model)

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
        nnzJ = self.model.nnzj + m
        J = psp(nrow=m, ncol=n, sizeHint=nnzJ)

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
        J.put(-1.0, lowerC, on + rlowerC)
        J.put(-1.0, upperC, on + nlowerC + rupperC)
        J.put(-1.0, rangeC, on + nlowerC + nupperC + rrangeC)

        return J

    def hess(self, x, z=None, obj_num=0, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        if z is None:
            z = np.zeros(self.m)

        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n

        H = psp(nrow=self.n, ncol=self.n, symmetric=True,
                sizeHint=self.model.nnzh)
        H[:on, :on] = model.hess(x[:on], z, obj_num, *args, **kwargs)
        return H
