#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for TRON."""

import logging
import sys
from nlp.model.cysparsemodel import AmplModel
from nlp.optimize.tron import TRON
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger

nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")

# Create TRON logger.
tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%12s %5s %5s %8s %7s %5s %5s %5s %s",
                "name", "nvar", "iter", "f", u"‖P∇f‖", "#f", u"#∇f", "stat",
                "time")

for problem in sys.argv[1:]:
    model = AmplModel(problem)
    model.compute_scaling_obj()

    # Check for inequality- or equality-constrained problem.
    if model.m > 0:
        msg = '%s has %d linear or nonlinear constraints'
        logger.error(msg, model.name, model.m)
        continue

    tron = TRON(model, TruncatedCG, maxiter=100)
    tron.solve()

    logger.info("%12s %5d %5d %8.1e %7.1e %5d %5d %5s %.3f",
                model.name, model.nvar, tron.iter, tron.f, tron.pgnorm,
                model.obj.ncalls, model.grad.ncalls,
                tron.status, tron.tsolve)
