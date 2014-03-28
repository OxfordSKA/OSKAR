# 
#  This file is part of OSKAR.
#
# TODO licence
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
    """DOC GOES HERE"""
    image_lib.make(uu,vv,amp,freq,fov,size)


    

