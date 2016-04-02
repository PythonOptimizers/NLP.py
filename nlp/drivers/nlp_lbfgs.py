#!/usr/bin/env python
"""Simple AMPL driver for L-BFGS."""

from nlp.model.amplpy import QNAmplModel
from nlp.optimize.lbfgs import LBFGS
from nlp.tools.logs import config_logger

from pykrylov.linop import InverseLBFGSOperator
import sys

if len(sys.argv) == 1:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
log = config_logger("nlp.lbfgs", "%(name)-9s %(levelname)-5s %(message)s")

model = QNAmplModel(sys.argv[1], H=InverseLBFGSOperator, scaling=True)
lbfgs = LBFGS(model)
lbfgs.solve()
