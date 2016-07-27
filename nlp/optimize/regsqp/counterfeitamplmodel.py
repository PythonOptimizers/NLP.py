from nlp.model.pysparsemodel import PySparseAmplModel
from pykrylov.linop import LinearOperator
import numpy as np


class CounterFeitAmplModel(PySparseAmplModel):
    """Define `jop` as a linear operator involving `jprod` and `jtprod`.

    Ampl doesn't define `jprod` or `jtprod`.
    They are here define as a product of a `PySparse` matrix Jacobian or its
    transpose and a vector.
    """

    def jop(self, x):
        """Obtain Jacobian at x as a linear operator."""
        return LinearOperator(self.n, self.m,
                              lambda v: self.jprod(x, v),
                              matvec_transp=lambda u: self.jtprod(x, u),
                              symmetric=False,
                              dtype=np.float)

    def jprod(self, x, p, **kwargs):
        """Evaluate Jacobian-vector product at x with p."""
        return self.jac(x, **kwargs) * p

    def jtprod(self, x, p, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        return p * self.jac(x, **kwargs)
