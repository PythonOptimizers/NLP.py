#!/usr/bin/env python
"""
NLPy: A Nonlinear Programming Environment in Python

NLPy is a programming environment that facilitates construction of optimization
algorithms by supplying a library of efficient building blocks.
D. Orban <dominique.orban@gerad.ca>
"""
import os
import sys

from setuptools import setup   # enables 'python setup.py develop'
from distutils.extension import Extension
import ConfigParser

import numpy as np

DOCLINES = __doc__.split("\n")

nlpy_config = ConfigParser.SafeConfigParser()
nlpy_config.read('site.cfg')


nlpy_ext = []
########
# AMPL #
########

# Read relevant NLPy-specific configuration options.
libampl_dir = nlpy_config.get('ASL', 'asl_dir')  # .split(os.pathsep)
libampl_libdir = os.path.join(libampl_dir, 'lib')
libampl_include = os.path.join(libampl_dir, os.path.join('include', 'asl'))

amplpy_params = {}
amplpy_params['library_dirs'] = [libampl_libdir]
amplpy_params['include_dirs'] = [os.path.join('nlpy', 'model', 'src'),
                                 libampl_include,
                                 np.get_include()]
amplpy_params['libraries'] = ['asl']

amplpy_src = [os.path.join('nlpy', 'model', 'src', '_amplpy.c'),
              os.path.join('nlpy', 'model', 'src', 'amplutils.c')]

nlpy_ext.append(Extension(name="nlpy.model._amplpy",
                sources=amplpy_src,
                **amplpy_params))


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

packages_list = ['nlpy',
                 'nlpy.model',
                 'nlpy.tools',
                 'tests']

scripts_list = [os.path.join('nlpy', 'tools', 'nlpy_pprof.py')]

setup(
    name='nlpy',
    version="0.0.1",
    author="Dominique Orban",
    author_email="dominique.orban@gmail.com",
    maintainer="NLPy Developers",
    maintainer_email="dominique.orban@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://github.com/PythonOptimizers/NLPy2.0.git",
    download_url="https://github.com/PythonOptimizers/NLPy2.0.git",
    license='LGPL',
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=filter(None, CLASSIFIERS.split('\n')),
    install_requires=['numpy'],
    ext_modules=nlpy_ext,
    package_dir={"nlpy": "nlpy"},
    packages=packages_list,
    scripts=scripts_list,
    zip_safe=False
    )
