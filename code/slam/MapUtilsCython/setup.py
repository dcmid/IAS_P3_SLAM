#!/usr/bin/env python3

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
        name = "MapUtils",
        ext_modules = cythonize("MapUtils_fclad.pyx", annotate=True)
    )

setup(
        name = "update_ogm",
        ext_modules = cythonize("update_ogm.pyx", annotate=True),
        include_dirs=[np.get_include()]
)