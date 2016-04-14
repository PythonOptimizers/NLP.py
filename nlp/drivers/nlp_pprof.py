#!/usr/bin/env python
"""Main driver for performance profiles."""

from nlp.tools.pprof import PerformanceProfile
from optparse import OptionParser


usage_msg = """%prog [options] file1 file2 [... fileN]
where file1 through fileN contain the statistics of a solver."""

# Define allowed command-line options.
parser = OptionParser(usage=usage_msg)
parser.add_option('-c', '--column', action='store', type='int', default=2,
                  dest='datacol', help='column containing the metrics')
parser.add_option('-l', '--linear', action='store_false', dest='logscale',
                  default=True, help='Use linear scale for x axis')
parser.add_option('-s', '--sep', action='store', type='string', dest='sep',
                  default=r'\s+', help='column separator (as a regexp)')
parser.add_option('-t', '--title', action='store', type='string', dest='title',
                  default='Deathmatch', help='plot title')
parser.add_option('-b', '--bw', action='store_true', dest='bw', default=False,
                  help='plot in shades of gray')

# Parse command-line options.
(options, args) = parser.parse_args()

pprof = PerformanceProfile(args, **options.__dict__)
pprof.plot()
