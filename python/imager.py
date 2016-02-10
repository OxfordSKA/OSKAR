# 
#  This file is part of OSKAR.
# 
# Copyright (c) 2014-2016, The University of Oxford
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
imager.py : OSKAR imager functions
=====================================================

This module provides an interface to the OSKAR imager.

Image creation
----------------------------------

- :func:`make_image` makes an image of visibility data

"""

import math
import _imager_lib

__all__ = ['make_image']

def make_image(uu, vv, ww, amp, fov, size):
    """make_image(uu, vv, ww, amp, fov, size)
    
    Makes an image from visibility data.
    
    Parameters
    ----------
    uu : array like, shape (n,), float64
        Input baseline uu coordinates, in wavelengths.
    vv : array like, shape (n,), float64
        Input baseline vv coordinates, in wavelengths.
    ww : array like, shape (n,), float64
        Input baseline ww coordinates, in wavelengths.
    amp : array like, shape (n,), complex128
        Input baseline amplitudes.
    fov : scalar, float64
        Image field of view, in degrees.
    size : scalar, int
        Image size along one dimension, in pixels.
    """
    return _imager_lib.make_image(uu, vv, ww, amp, fov, size)


def fov_to_cellsize(fov_rad, size):
    """
    fov_to_cellsize(fov_rad, size)
    
    Convert image FoV and size along one dimension in pixels to cellsize.

    Arguments:
    fov_deg -- Image FoV, in radians
    size    -- Image size in one dimension in pixels

    Return:
    Image cellsize, in radians.
    """
    rmax = math.sin(0.5 * fov_rad)
    inc = 2.0 * rmax / size
    return math.asin(inc)

