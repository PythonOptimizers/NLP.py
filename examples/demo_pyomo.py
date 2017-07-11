# -*- coding: utf-8 -*-
try:
    from pyomo.environ import Var, Objective, ConcreteModel
    from pyomo.opt import ProblemFormat
except ImportError:
    import sys
    print "Pyomo is not installed."
    sys.exit(0)

from nlp.model.amplmodel import AmplModel
from nlp.optimize.tron import TRON
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger
from nlp.tools.utils import evaluate_model_methods_at_starting_point
import logging

import numpy as np

tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.INFO)


if __name__ == "__main__":
    # Create a Pyomo Rosenbrock model
    pyomo_model = ConcreteModel()
    pyomo_model.x = Var()
    pyomo_model.y = Var(bounds=(-1.5, None))
    pyomo_model.o = Objective(expr=(pyomo_model.x - 1)**2 + \
                                    100 * (pyomo_model.y - pyomo_model.x**2)**2)
    pyomo_model.x.set_value(-2.)
    pyomo_model.y.set_value(1.)

    # Writes a Pyomo model in NL file format."""
    nl_filename = "rosenbrock.nl"
    _, smap_id = pyomo_model.write(nl_filename,
                             format=ProblemFormat.nl)

    # Create an Amplmodel from the generated nl file
    nlp_model = AmplModel(nl_filename)
    evaluate_model_methods_at_starting_point(nlp_model)

    solver = TRON(nlp_model, TruncatedCG)
    solver.solve()

