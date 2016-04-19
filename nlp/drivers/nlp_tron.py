#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for TRON."""

import logging
import sys
from argparse import ArgumentParser

from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger


def tron_stats(tron):
    """Obtain TRON statistics and indicate failures with negatives."""
    if tron.status in ("fatol", "frtol", "gtol"):
        it = tron.iter
        fc, gc = tron.model.obj.ncalls, tron.model.grad.ncalls
        pg = tron.pgnorm
        ts = tron.tsolve
    else:
        it = -tron.iter
        fc, gc = -tron.model.obj.ncalls, -tron.model.grad.ncalls
        pg = -1.0 if tron.pgnorm is None else -tron.pgnorm
        ts = -1.0 if tron.tsolve is None else -tron.tsolve
    return (it, fc, gc, pg, ts)


desc = """Trust-region active-set method for bound-constrained problems based
on TRON. By default, exact second derivatives are used."""

# Define allowed command-line options.
parser = ArgumentParser(description=desc)
parser.add_argument("-1", "--sr1", action="store_true", dest="sr1",
                    default=False, help="use limited-memory SR1 approximation")
parser.add_argument("-2", "--bfgs", action="store_true", dest="bfgs",
                    default=False,
                    help="use limited-memory BFGS approximation")
parser.add_argument("-p", "--pairs", type=int,
                    default=5, dest="npairs", help="quasi-Newton memory")
parser.add_argument("-b", "--backtrack", action="store_true", dest="ny",
                    default=False,
                    help="backtrack along rejected trust-region step")
parser.add_argument("-i", "--iter", type=int,
                    default=100, dest="maxiter",
                    help="maximum number of iterations")

# Parse command-line arguments.
(args, other) = parser.parse_known_args()
opts = {}

# Import appropriate components.
if args.sr1 or args.bfgs:
    from nlp.model.amplmodel import QNAmplModel as Model
    from nlp.optimize.tron import QNTRON as TRON
    if args.sr1:
        from pykrylov.linop import CompactLSR1Operator as QNOperator
    else:
        from pykrylov.linop import CompactLBFGSOperator as QNOperator
    opts["H"] = QNOperator
    opts["npairs"] = args.npairs
    opts["scaling"] = True
else:
    from nlp.model.amplmodel import AmplModel as Model
    from nlp.optimize.tron import TRON

nprobs = len(other)
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")

# Create TRON logger.
tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)

logger.info("%12s %5s %6s %8s %8s %6s %6s %5s %7s",
            "name", "nvar", "iter", "f", u"‖P∇f‖",
            "#f", u"#∇f", "stat", "time")

for problem in other:
    model = Model(problem, **opts)
    model.compute_scaling_obj()

    # Check for inequality- or equality-constrained problem.
    if model.m > 0:
        msg = '%s has %d linear or nonlinear constraints'
        logger.error(msg, model.name, model.m)
        continue

    tron = TRON(model, TruncatedCG, maxiter=args.maxiter, ny=args.ny)
    try:
        tron.solve()
        status = tron.status
        niter, fcalls, gcalls, pgnorm, tsolve = tron_stats(tron)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, pgnorm, tsolve = tron_stats(tron)

    logger.info("%12s %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
                model.name, model.nvar, niter, tron.f, pgnorm,
                fcalls, gcalls, status, tsolve)
