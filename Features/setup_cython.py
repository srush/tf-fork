from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
ext_modules = [Extension("fast_inside_outside", ["fast_inside_outside.pyx"])]

setup(
    name = 'Fast Inside-Outside app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()]  
    )
