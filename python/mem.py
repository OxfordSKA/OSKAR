# 
#  This file is part of OSKAR.
# 
# Copyright (c) 2014, The University of Oxford
# All rights reserved.
#
#  This file is part of the OSKAR package.
#  Contact: oskar at oerc.ox.ac.uk
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  3. Neither the name of the University of Oxford nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
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
    char    = 0x01
    int     = 0x02
    single  = 0x04
    double  = 0x08
    complex = 0x20
    matrix  = 0x40
    single_complex = single | complex
    double_complex = double | complex
    single_complex_matrix = single | complex | matrix
    double_complex_matrix = double | complex | matrix

class oskar_Mem:

    def __init__(self, length, type=mtype.double, location=mloc.cpu):
        (self.__handle,_) = mem_lib.create(type, location, length)

    def location(self):
        return mem_lib.location(self.__handle)



