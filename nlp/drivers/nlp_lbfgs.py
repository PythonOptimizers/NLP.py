#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for L-BFGS."""

import logging
import sys
from argparse import ArgumentParser
from nlp.model.amplmodel import QNAmplModel
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

desc = """Linesearch-based limited-memory BFGS method in inverse form."""

# Define allowed command-line options.
parser = ArgumentParser(description=desc)
parser.add_argument("-p", "--pairs", type=int,
                    default=5, dest="npairs", help="BFGS memory")
parser.add_argument("-a", "--armijo", action="store_true", dest="armijo",
                    default=False, help="use improved Armijo linesearch")
parser.add_argument("-i", "--iter", type=int,
                    default=100, dest="maxiter",
                    help="maximum number of iterations")

# Parse command-line arguments.
(args, other) = parser.parse_known_args()

if args.armijo:
    from nlp.optimize.lbfgs import LBFGS
else:
    from nlp.optimize.lbfgs import WolfeLBFGS as LBFGS

nprobs = len(other)
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

for problem in other:
    model = QNAmplModel(problem,
                        H=InverseLBFGSOperator,
                        npairs=args.npairs,
                        scaling=True)
    model.compute_scaling_obj()

    lbfgs = LBFGS(model, maxiter=args.maxiter)
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
