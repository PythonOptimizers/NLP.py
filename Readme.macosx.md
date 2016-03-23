# Build instructions for Mac OS/X users

Installing NLP.py will be much easier if you use [Homebrew](https://brew.sh).
Follow the instructions to install Homebrew.
Then, the following dependencies can be installed automatically:

```bash
brew install gcc  # currently v5. Contains gfortran

brew tap homebrew/science
brew install adol-c                             # will also install Colpack
brew install boost --with-mpi --without-single  # to use pycppad
brew install cppad --with-adol-c --with-boost --cc=gcc-5
brew install asl
brew install metis

pip install algopy
pip install git+https://github.com/b45ch1/pycppad.git
```

## Installing PyAdolc

```bash
git clone https://github.com/b45ch1/pyadolc.git
cd pyadolc
BOOST_DIR=$(brew --prefix boost-python) ADOLC_DIR=$(brew --prefix adol-c) COLPACK_DIR=$(brew --prefix colpack) CC=clang CXX=clang++ python setup.py install
cd
python -c "import adolc; adolc.test()"
```

If you encounter build errors, edit and change `setup.py` as follows:
```diff
diff --git a/setup.py b/setup.py
index 5e6e695..68f5c68 100644
--- a/setup.py
+++ b/setup.py
@@ -36,7 +36,7 @@ colpack_lib_path1    = os.path.join(COLPACK_DIR, 'lib')
 colpack_lib_path2    = os.path.join(COLPACK_DIR, 'lib64')

 # ADAPT THIS TO FIT YOUR SYSTEM
-extra_compile_args = ['-std=c++11 -ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB']
+extra_compile_args = ['-std=c++11 -stdlib=libc++ -mmacosx-version-min=10.9 -ftemplate-depth-100 -DBOOST_PYTHON_DYNAMIC_LIB']
 include_dirs = [get_numpy_include_dirs()[0], boost_include_path, adolc_include_path, colpack_include_path]
 library_dirs = [boost_library_path1, boost_library_path2, adolc_library_path1, adolc_library_path2, colpack_lib_path1, colpack_lib_path2]
 libraries = ['boost_python','adolc', 'ColPack']
```
