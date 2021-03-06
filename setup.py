#!/usr/bin/env python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        'rk1',
        ['rk1.pyx'],
        extra_compile_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
        #define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]


setup(
    ext_modules=cythonize(ext_modules)
)
