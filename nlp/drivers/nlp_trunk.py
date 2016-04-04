#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for TRUNK."""

import logging
import sys
from nlp.model.amplpy import AmplModel
from nlp.tr.trustregion import TrustRegion
from nlp.optimize.trunk import Trunk
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger

nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp",
                       "%(name)-3s %(levelname)-5s %(message)s")

# Create TRUNK logger.
slv_log = config_logger("nlp.trunk",
                        "%(name)-9s %(levelname)-5s %(message)s",
                        level=logging.WARN if nprobs > 1 else logging.INFO)

logger.info("%10s %5s %8s %7s %5s %5s %4s %s",
            "name", "nvar", "f", u"‖∇f‖", "#f", u"#∇f", "stat", "time")
for problem in sys.argv[1:]:
    model = AmplModel(problem)
    trunk = Trunk(model, TrustRegion(), TruncatedCG,
                  ny=True, inexact=True, maxiter=500)
    trunk.solve()
    logger.info("%10s %5d %8.1e %7.1e %5d %5d %4s %.3f",
                model.name, model.nvar, trunk.f, trunk.gNorm,
                model.obj.ncalls, model.grad.ncalls,
                trunk.status, trunk.tsolve)
