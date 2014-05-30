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
vis.py : OSKAR visibility related functions
=====================================================

This module provides functions related to OSKAR visibility data.

- :func:`read` reads an oskar visibility binary file

"""

import numpy as np
import exceptions
import _vis_lib as vis_lib

__all__ = ['']

class oskar_Vis:

    def __init__(self, filename):
        (self.__handle,_) = vis_lib.read(filename)

    def num_baselines(self):
        return vis_lib.num_baselines(self.__handle)

    def num_channels(self):
        return vis_lib.num_channels(self.__handle)

    def num_times(self):
        return vis_lib.num_times(self.__handle)

    def station_coords(self):
        return vis_lib.station_coords(self.__handle)

    def lon(self):
        return vis_lib.lon(self.__handle)

    def lat(self):
        return vis_lib.lat(self.__handle)

    def baseline_coords(self, hermitian_copy=False):
        import numpy as np
        (uu,vv,ww) = vis_lib.baseline_coords(self.__handle)
        if hermitian_copy == True:
            uu = np.concatenate((uu,-uu), axis=0)
            vv = np.concatenate((vv,-vv), axis=0)
            ww = np.concatenate((ww,-ww), axis=0)
        return (uu,vv,ww)

    def linear_amps(self):
        a = vis_lib.amplitude(self.__handle)
        return (a[:,0],a[:,1],a[:,2],a[:,3]) 

    def stokes_amps(self):
        import numpy as np
        a = vis_lib.amplitude(self.__handle)
        I = 0.5 * (a[:,0] + a[:,3])
        Q = 0.5 * (a[:,0] - a[:,3])
        U = 0.5 * (a[:,1] + a[:,2])
        V = -0.5j * (a[:,1] - a[:,2])
        return (I,Q,U,V)



