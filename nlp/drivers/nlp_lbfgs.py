#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for L-BFGS."""

import logging
import sys
from nlp.model.amplmodel import QNAmplModel
from nlp.optimize.lbfgs import LBFGS, WolfeLBFGS
from nlp.tools.logs import config_logger

from pykrylov.linop import InverseLBFGSOperator


def lbfgs_stats(lbfgs):
    """Obtain L-BFGS statistics and indicate failures with negatives."""
    if lbfgs.status == "opt":
        it = lbfgs.iter
        fc, gc = lbfgs.model.obj.ncalls, lbfgs.model.grad.ncalls
        gn = lbfgs.g_norm
        ts = lbfgs.tsolve
    else:
        it = -lbfgs.iter
        fc, gc = -lbfgs.model.obj.ncalls, -lbfgs.model.grad.ncalls
        gn = -1.0 if lbfgs.g_norm is None else -lbfgs.g_norm
        ts = -1.0 if lbfgs.tsolve is None else -lbfgs.tsolve
    return (it, fc, gc, gn, ts)


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

logger.info("%10s %5s %6s %8s %8s %6s %6s %5s %7s",
            "name", "nvar", "iter", "f", u"‖∇f‖", "#f", u"#∇f", "stat", "time")

for problem in sys.argv[1:]:
    model = QNAmplModel(problem, H=InverseLBFGSOperator, scaling=True)
    model.compute_scaling_obj()

    lbfgs = WolfeLBFGS(model, maxiter=300)
    try:
        lbfgs.solve()
        status = lbfgs.status
        niter, fcalls, gcalls, gnorm, tsolve = lbfgs_stats(lbfgs)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, gnorm, tsolve = lbfgs_stats(lbfgs)

    logger.info("%10s %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
                model.name, model.nvar, niter, lbfgs.f, gnorm,
                fcalls, gcalls, status, tsolve)
