"""Example usage of AdolcModel."""

import numpy as np

from nlp.model.adolcmodel import BaseAdolcModel, SciPyAdolcModel


class AdolcRosenbrock(BaseAdolcModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        """Rosenbrock objective value."""
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class SciPyRosenbrock(SciPyAdolcModel, AdolcRosenbrock):
    """The Rosenbrock problem with sparse matrices in SciPy format."""

    pass


n = 5

# The following model has exact second derivatives in sparse SciPy format.
scipy_model = SciPyRosenbrock(n, name="Rosenbrock", x0=-np.ones(n))
