#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for Funnel."""

import sys
import logging
from argparse import ArgumentParser

from nlp.tools.logs import config_logger

desc = """A trust-funnel method for equality-constrained problems based
on Funnel. By default, exact second derivatives are used."""

# Define allowed command-line options.
parser = ArgumentParser(description=desc)
parser.add_argument("-1", "--sr1", action="store_true", dest="sr1",
                    default=False, help="use limited-memory SR1 approximation")
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
if args.sr1:
    from nlp.model.pysparsemodel import QnPySparseAmplModel as Model
    from nlp.optimize.funnel import QNFunnel as Funnel
    from pykrylov.linop import CompactLSR1Operator as QNOperator

    opts["H"] = QNOperator
    opts["npairs"] = args.npairs
    opts["scaling"] = True
else:
    from nlp.model.pysparsemodel import PySparseAmplModel as Model
    from nlp.optimize.funnel import Funnel

nprobs = len(other)
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-9s %(levelname)-5s %(message)s")

# Create Funnel logger.
funnel_logger = config_logger("funnel",
                              "%(name)-9s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.DEBUG)
qn_logger = config_logger("qn",
                          "%(name)-9s %(levelname)-5s %(message)s",
                          level=logging.WARN if nprobs > 1 else logging.DEBUG)

if nprobs > 1:
    logger.info("%12s %5s %5s %8s %8s %8s %6s %6s %6s %7s",
                "name", "nvar", "iter", "f", u"‖c‖", u"‖g+Jᵀy‖", "#f", u"#∇f",
                "stat", "time")

for problem in other:
    model = Model(problem, logger=qn_logger, **opts)

    # Check for inequality-constrained problem.
    if model.nlowerC > 0 or model.nupperC > 0 or model.nrangeC > 0:
        msg = '%s has %d inequality constraints'
        logger.error(msg, model.name, model.nlowerC +
                     model.nupperC + model.nrangeC)
        error = True
        continue

    # Check for bound-constrained problem.
    if model.nbounds > 0:
        msg = '%s has %d bound constraints'
        logger.error(msg, model.name, model.nbounds)
        error = True
        continue

    funnel = Funnel(model, maxiter=args.maxiter, logger_name="funnel")
    funnel.solve()

    if nprobs > 1:
        logger.info("%12s %5d %5d %8.1e %8.1e %8.1e %6d %6d %6s %7.3f",
                    model.name, model.nvar, funnel.iter, funnel.f,
                    funnel.p_resid, funnel.d_resid, model.obj.ncalls,
                    model.grad.ncalls, funnel.status, funnel.tsolve)
    else:
        # Output final statistics
        logger.info('--------------------------------')
        logger.info('Funnel: End of Execution')
        logger.info('  Problem                      : %-s', model.name)
        logger.info('  Number of variables          : %-d', model.nvar)
        logger.info('  Initial/Final Objective      : %-g/%-g',
                    funnel.f0, funnel.f)
        logger.info('  Number of iterations         : %-d', funnel.iter)
        logger.info('  Number of function evals     : %-d', model.obj.ncalls)
        logger.info('  Number of gradient evals     : %-d', model.grad.ncalls)
        logger.info('  Solve time                   : %-gs', funnel.tsolve)
        logger.info('  Status                       : %-s', funnel.status)
        logger.info('--------------------------------')
