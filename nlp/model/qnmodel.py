"""Models with quasi-Newton Hessian approximation."""

from nlp.model.nlpmodel import NLPModel

__docformat__ = 'restructuredtext'


class QuasiNewtonModel(NLPModel):
    """`NLPModel with a quasi-Newton Hessian approximation."""

    def __init__(self, *args, **kwargs):
        """Instantiate a model with quasi-Newton Hessian approximation.

        :keywords:
            :H: the `class` of a quasi-Newton linear operator.
                This keyword is mandatory.

        Keywords accepted by the quasi-Newton class will be passed
        directly to its constructor.
        """
        super(QuasiNewtonModel, self).__init__(*args, **kwargs)
        qn_cls = kwargs.pop('H')
        self._H = qn_cls(self.nvar, **kwargs)

    @property
    def H(self):
        """Quasi-Newton Hessian approximation."""
        return self._H

    def hess(self, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z).

        This is an alias for `self.H`.
        """
        return self.H

    def hop(self, *args, **kwargs):
        """Obtain Lagrangian Hessian at (x, z) as a linear operator.

        This is an alias for `self.H`.
        """
        return self.H

    def hprod(self, x, z, v, **kwargs):
        """Hessian-vector product."""
        return self.H * v
