# -*- coding: utf-8 -*-
"""Demo of the limited-memory BFGS method.

Each problem given on the command line is solved
for several values of the limited-memory parameter.
"""

from nlp.model.amplmodel import QNAmplModel
from nlp.optimize.lbfgs import LBFGS
from pykrylov.linop import InverseLBFGSOperator
from os.path import basename, splitext
import sys

headerfmt = "%-15s %-6s %-5s %-8s %-7s %-5s %-5s\n"
header = headerfmt % ("problem", "nvar", "pairs", "f", u"‖∇f‖", "iter", "time")
format = "%-15s %-6d %-5d %-8.1e %-7.1e %-5d %-5.2f\n"
sys.stdout.write(header)

for problem_name in sys.argv[1:]:
    for m in [1, 2, 3, 4, 5, 10, 15, 20]:
        model = QNAmplModel(problem_name,
                            H=InverseLBFGSOperator, scaling=True, npairs=m)

        lbfgs = LBFGS(model)
        lbfgs.solve()

        # Output final statistics
        probname = basename(splitext(problem_name)[0])
        sys.stdout.write(format % (probname, model.n, model.H.npairs, lbfgs.f,
                                   lbfgs.gNorm, lbfgs.iter, lbfgs.tsolve))
