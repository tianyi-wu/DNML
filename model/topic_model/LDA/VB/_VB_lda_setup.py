# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("_VB_lda", ["_VB_lda.pyx"], include_dirs=[numpy.get_include()])]
)
