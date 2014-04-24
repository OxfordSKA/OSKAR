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
image.py : OSKAR image related functions
=====================================================

This module provides functions related to OSKAR images.

Image creation
----------------------------------

- :func:`make` makes an image of visibility data by DFT on the GPU

"""

import numpy as np
import exceptions
import _image_lib as image_lib

__all__ = ['make']

def make(uu,vv,amp,freq,fov,size):
    """make(uu,vv,amp,freq,fov=2.0,size=128)
    
    Makes an image from visibility data. Computation is performed using a 
    DFT implemented on the GPU using CUDA.
    
    Parameters
    ----------
    uu : array like, shape (n,), float64
        Input baseline uu coordinates, in metres.
    vv : array like, shape (n,), float64
        Input baseline vv coordinates, in metres.
    amp : array like, shape (n,), complex128
        Input baseline amplitudes.
    freq : scalar, float64
        Frequency, in Hz.
    fov : scalar, default = 2.0
        Image field of view, in degrees.
    size : integer, default=128
        Image size along one dimension, in pixels.
    """
    return image_lib.make(uu,vv,amp,freq,fov,size)


def fov_to_cellsize(fov_deg, size):
    """
    fov_to_cellsize(fov_deg, size)
    
    Convert image FoV and size along one dimension in pixels to cellsize in arcseconds.

    Arguments:
    fov_deg -- Image FoV, in degrees
    size    -- Image size in one dimension in pixels

    Return:
    Image cellsize, in arcseconds
    """
    import numpy as np
    rmax = np.sin(fov_deg/2.0*(np.pi/180.0))
    inc = rmax / (0.5 * size)
    return np.arcsin(inc)*((180.0*3600.0)/np.pi)

    

