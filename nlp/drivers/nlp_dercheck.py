#!/usr/bin/env python
"""Simple AMPL driver for the derivative checker."""

from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.tools.dercheck import DerivativeChecker
from nlp.tools.logs import config_logger

if len(sys.argv) == 1:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
log = config_logger("nlp.der", "%(name)-10s %(levelname)-8s %(message)s")

nlp = PySparseAmplModel(sys.argv[1])
dcheck = DerivativeChecker(nlp, nlp.x0)
dcheck.check()
