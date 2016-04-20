#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for Funnel."""

import sys
import logging

from nlp.optimize.funnel import Funnel
from nlp.tools.logs import config_logger
from nlp.model.pysparsemodel import PySparseAmplModel

nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-9s %(levelname)-5s %(message)s")

# Create Funnel logger.
funnel_logger = config_logger("funnel",
                              "%(name)-9s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.DEBUG)
if nprobs > 1:
    logger.info("%12s %5s %6s %8s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "iter", "f", u"‖c‖", u"‖g+Jᵀy‖", "#f", u"#∇f",
                "stat", "time")

for problem in sys.argv[1:]:
    model = PySparseAmplModel(problem)

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

    funnel = Funnel(model, logger_name="funnel")
    funnel.solve()

    if nprobs > 1:
        logger.info("%12s %5d %5d %8.1e %8.1e %8.1e %5d %5d %4s %.3f",
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
