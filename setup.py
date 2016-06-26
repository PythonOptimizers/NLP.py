#!/usr/bin/env python
"""NLP.py: A Nonlinear Programming Environment in Python.

NLP.py is a programming environment that facilitates construction of
optimization algorithms by supplying a library of efficient building blocks.

D. Orban     <dominique.orban@gerad.ca>
S. Arreckx   <sylvain.arreckx@gmail.com>
"""
import os
import glob

from setuptools import setup   # enables 'python setup.py develop'
from distutils.extension import Extension
import ConfigParser

import numpy as np

DOCLINES = __doc__.split("\n")

nlp_config = ConfigParser.SafeConfigParser()
nlp_config.read('site.cfg')


nlp_ext = []

# Optional AMPL extension
if nlp_config.has_section('ASL'):
    # Read relevant AMPL-specific configuration options.
    libampl_dir = nlp_config.get('ASL', 'asl_dir')  # .split(os.pathsep)
    libampl_libdir = os.path.join(libampl_dir, 'lib')
    libampl_include = os.path.join(libampl_dir, os.path.join('include', 'asl'))

    if not (os.path.isdir(libampl_libdir) and os.path.isdir(libampl_include)):
        raise ValueError("Please check if ASL paths are correct:" +
                         "\n   %s\n    %s" % (libampl_libdir, libampl_include))

    amplmodel_params = {}
    amplmodel_params['library_dirs'] = [libampl_libdir]
    amplmodel_params['include_dirs'] = [os.path.join('nlp', 'model', 'src'),
                                        libampl_include,
                                        np.get_include()]
    amplmodel_params['libraries'] = ['asl']

    amplmodel_src = [os.path.join('nlp', 'model', 'src', '_amplmodel.c'),
                     os.path.join('nlp', 'model', 'src', 'amplutils.c')]

    nlp_ext.append(Extension(name="nlp.model._amplmodel",
                             sources=amplmodel_src,
                             **amplmodel_params))


#############################
# Strong Wolfe linesearches #
#############################
swls_params = {}
swls_params['include_dirs'] = [np.get_include()]

swls_src = [os.path.join('nlp', 'ls', 'src', '_strong_wolfe_linesearch.c')]

nlp_ext.append(Extension(name="nlp.ls._strong_wolfe_linesearch",
                         sources=swls_src,
                         **swls_params))

mswls_params = {}
mswls_params['include_dirs'] = [np.get_include()]

mswls_src = [os.path.join('nlp', 'ls', 'src',
                          '_modified_strong_wolfe_linesearch.c')]

nlp_ext.append(Extension(name="nlp.ls._modified_strong_wolfe_linesearch",
                         sources=mswls_src,
                         **mswls_params))

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: LGPL
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

packages_list = ['nlp',
                 'nlp.model',
                 'nlp.ls',
                 'nlp.optimize',
                 'nlp.tools',
                 'nlp.tr']

scripts_list = glob.glob(os.path.join('nlp', 'drivers', 'nlp_*.py'))

setup(
    name='nlp',
    version="0.0.1",
    author="Dominique Orban",
    author_email="dominique.orban@gmail.com",
    maintainer="NLP.py Developers",
    maintainer_email="dominique.orban@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://github.com/PythonOptimizers/NLP.py.git",
    download_url="https://github.com/PythonOptimizers/NLP.py.git",
    license='LGPL',
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=filter(None, CLASSIFIERS.split('\n')),
    install_requires=['numpy'],
    ext_modules=nlp_ext,
    package_dir={"nlp": "nlp"},
    packages=packages_list,
    scripts=scripts_list,
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    zip_safe=False
)
