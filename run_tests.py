"""
Run all or some tests.

By default, we use 'nosetests' but 'unittest discover' can be used instead. See help.

"""
from __future__ import print_function

import os
import sys
import subprocess
import shutil
import distutils
import argparse

try:
    import nose
except ImportError:
    print("You need to install nose to run the tests.")
    sys.exit(-1)


def make_parser():
    """
    Create a comment line argument parser.

    Returns:
        The command line parser.
    """
    parser = argparse.ArgumentParser(description='%s: run all or some tests for the CySparse library' % os.path.basename(sys.argv[0]))
    parser.add_argument("-r", "--rebuild", help="Rebuild from scratch the CySparse library", action='store_true', required=False)
    parser.add_argument("-b", "--build", help="Build (if needed) CySparse library with new code", action='store_true', required=False)
    parser.add_argument("-p", "--pattern", help="Run tests with a given filename pattern", required=False)
    parser.add_argument("-v", "--verbose", help="Add some context on the console", action='store_true', required=False)
    parser.add_argument("-n", "--dont_use_nose", help="Use unittest discover instead of nosetests", action='store_true', required=False)

    return parser


def generate_lib():
    subprocess.call(['python', 'setup.py', 'build'])


def launch_nosetests(pattern=None, verbose=False, use_libraries=None):
    current_dir = os.getcwd()
    os.chdir(lib_dir)

    commands_list = ['nosetests']
    if verbose:
        commands_list.append('--verbosity=2')

    if pattern is not None:
        commands_list.append('-p')
        commands_list.append(pattern)

    commands_list.append('tests')

    if verbose:
        print("launch command: '%s':" % " ".join(commands_list))
    subprocess.call(commands_list)

    os.chdir(current_dir)


def launch_unittest(pattern=None, verbose=False):
    current_dir = os.getcwd()
    os.chdir(lib_dir)

    commands_list = ['python', '-m', 'unittest', 'discover', 'tests', '-c']

    if verbose:
        commands_list.append('-v')
    if pattern is not None:
        commands_list.append('-p')
        commands_list.append(pattern)

    if verbose:
        print("launch command: '%s':" % " ".join(commands_list))
    subprocess.call(commands_list)

    os.chdir(current_dir)

if __name__ == "__main__":

    # line arguments
    parser = make_parser()
    arg_options = parser.parse_args()

    platform = distutils.util.get_platform()
    python_version = sys.version

    lib_dir =  "build" + os.path.sep + "lib." + platform + "-" + python_version[0:3]
    destination_dir = lib_dir + os.path.sep + "tests"

    if arg_options.verbose:
        print("Deleting test directory %s... " % destination_dir,)
    # clean libxxx/tests because shutil.copytree only copies non existing directories
    shutil.rmtree(destination_dir, ignore_errors=True)
    if arg_options.verbose:
        print("done")
        print("copying tests into test directory %s..." % destination_dir,)
    shutil.copytree("tests", destination_dir, symlinks=False, ignore=None)
    if arg_options.verbose:
        print("done")

    if arg_options.verbose:
        print("Generating lib...",)
    generate_lib()
    if arg_options.verbose:
        print("done")


    if arg_options.dont_use_nose:
        launch_unittest(arg_options.pattern, arg_options.verbose)
    else:
        launch_nosetests(arg_options.pattern, arg_options.verbose)
