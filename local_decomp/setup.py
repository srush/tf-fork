from distutils.core import setup, Extension
import os.path
from Cython.Distutils import build_ext


srilm_prefix = "/home/srush/libs/sri"
srilm_include = srilm_prefix + "/include"
srilm_lib = srilm_prefix + "/lib/i686"

local_module = Extension('local_decomp',
                         language="c++",
                         
                         sources = [ 'Local.pyx', 'Bigram.cpp', 'GraphDecompose.cpp','Graph.cpp', 'LMCache.cpp', 'dual_subproblem.cpp','WordHolder.cpp'],
                         include_dirs = [srilm_include],
                         undef_macros=['NDEBUG'],
                         libraries = ["stdc++", 'oolm', 'dstruct', 'misc'],
                         library_dirs = [srilm_lib]
                         )

setup (name = 'LOCAL_DECOMP',
       version = '1.0',
       description = 'Interface ',
       ext_modules = [local_module],
       cmdclass = {'build_ext': build_ext}
       )

