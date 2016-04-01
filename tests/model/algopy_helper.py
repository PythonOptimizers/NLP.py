import algopy
from nlp.model.algopymodel import AlgopyModel


class AlgopyRosenbrock(AlgopyModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        return algopy.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class AlgopyHS7(AlgopyModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def obj(self, x, **kwargs):
        return algopy.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        c = algopy.zeros(1, dtype=x)

        # AlgoPy doesn't support cons_pos()
        c[0] = (1 + x[0]**2)**2 + x[1]**2 - 4
        return c
