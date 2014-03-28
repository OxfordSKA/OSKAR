# 
#  This file is part of OSKAR.
#
# TODO licence
# 
"""
=====================================================
mem.py : OSKAR mem related functions
=====================================================

This module provides functions related to OSKAR mem.


- :func:`create` creates an oskar mem structure handle.
- :func:`location` returns the location of an oskar mem structure.

"""

import numpy as np
import exceptions
import _mem_lib as mem_lib

__all__ = ['create']

class mloc:
    cpu = 0
    gpu = 1

class mtype:
    char = 1
    int = 2
    single = 4
    double = 8
    complex = 20
    matrix = 40
    single_complex = (single|complex)
    double_complex = (double|complex)
    single_complex_matrix = (single|complex|matrix)
    double_complex_matrix = (double|complex|matrix)

class oskar_Mem:

    def __init__(self, length, type=mtype.double, location=mloc.cpu):
        self.__handle = mem_lib.create(type, location, length)

    def location(self):
        return mem_lib.location(self.__handle)



