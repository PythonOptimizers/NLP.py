#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for L-BFGS."""

import logging
import sys
from nlp.model.amplmodel import QNAmplModel
from nlp.optimize.lbfgs import LBFGS
from nlp.tools.logs import config_logger

from pykrylov.linop import InverseLBFGSOperator

nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp",
                       "%(name)-3s %(levelname)-5s %(message)s")

# Create LBFGS logger.
slv_log = config_logger("nlp.lbfgs",
                        "%(name)-9s %(levelname)-5s %(message)s",
                        level=logging.WARN if nprobs > 1 else logging.INFO)

logger.info("%10s %5s %8s %7s %5s %5s %4s %s",
            "name", "nvar", "f", u"‖∇f‖", "#f", u"#∇f", "stat", "time")
for problem in sys.argv[1:]:
    model = QNAmplModel(problem, H=InverseLBFGSOperator, scaling=True)
    lbfgs = LBFGS(model, maxiter=300)
    lbfgs.solve()
    logger.info("%10s %5d %8.1e %7.1e %5d %5d %4s %.3f",
                model.name, model.nvar, lbfgs.f, lbfgs.gNorm,
                model.obj.ncalls, model.grad.ncalls,
                lbfgs.status, lbfgs.tsolve)
