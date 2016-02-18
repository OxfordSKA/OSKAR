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

class Imager(object):
    def __init__(self, precision="double"):
        self._capsule, _ = _imager_lib.create(precision)

    def finalise(self, image=None):
        _imager_lib.finalise(self._capsule, image)

    def reset_cache(self):
        _imager_lib.reset_cache(self._capsule)

    def run(self, filename):
        _imager_lib.run(self._capsule, filename)

    def set_algorithm(self, algorithm_type):
        _imager_lib.set_algorithm(self._capsule, algorithm_type)

    def set_default_direction(self):
        _imager_lib.set_default_direction(self._capsule)

    def set_direction(self, ra_deg, dec_deg):
        _imager_lib.set_direction(self._capsule, ra_deg, dec_deg)

    def set_channel_range(self, start, end, snapshots):
        _imager_lib.set_channel_range(self._capsule, start, end, snapshots)

    def set_grid_kernel(self, kernel_type, support, oversample):
        _imager_lib.set_grid_kernel(self._capsule, kernel_type, \
            support, oversample)

    def set_image_type(self, image_type):
        _imager_lib.set_image_type(self._capsule, image_type)

    def set_fov(self, fov_deg):
        _imager_lib.set_fov(self._capsule, fov_deg)

    def set_ms_column(self, column):
        _imager_lib.set_ms_column(self._capsule, column)

    def set_output_root(self, filename):
        _imager_lib.set_output_root(self._capsule, filename)

    def set_size(self, size):
        _imager_lib.set_size(self._capsule, size)

    def set_time_range(self, start, end, snapshots):
        _imager_lib.set_time_range(self._capsule, start, end, snapshots)

    def set_vis_frequency(self, ref_hz, inc_hz, num_channels):
        _imager_lib.set_vis_frequency(self._capsule, \
            ref_hz, inc_hz, num_channels)

    def set_vis_phase_centre(self, ra_deg, dec_deg):
        _imager_lib.set_vis_phase_centre(self._capsule, ra_deg, dec_deg)

    def set_vis_time(self, ref_mjd_utc, inc_sec, num_times):
        _imager_lib.set_vis_time(self._capsule, ref_mjd_utc, inc_sec, num_times)

    def update(self, num_baselines, uu, vv, ww, amps, weight, 
        num_pols = 1, start_time = 0, end_time = 0, 
        start_channel = 0, end_channel = 0):
        _imager_lib.update(self._capsule, num_baselines, uu, vv, ww, 
            amps, weight, num_pols, start_time, end_time, 
            start_channel, end_channel)

    @staticmethod
    def make_image(uu, vv, ww, amp, weight, fov, size):
        """
        make_image(uu, vv, ww, amp, weight, fov, size)
        
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
            Input baseline visibility amplitudes.
        weight : array like, shape (n,), float64
            Input visibility weights.
        fov : scalar, float64
            Image field of view, in degrees.
        size : scalar, int
            Image size along one dimension, in pixels.
        """
        return _imager_lib.make_image(uu, vv, ww, amp, weight, fov, size)

    @staticmethod
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

