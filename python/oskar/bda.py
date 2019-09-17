# -*- coding: utf-8 -*-
#
# Copyright (c) 2016, The University of Oxford
# All rights reserved.
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
=============
BDA functions
=============
"""
from __future__ import division, absolute_import, print_function
try:
    from . import _bda_utils
except ImportError:
    _bda_utils = None


class BDA(object):
    def __init__(self, num_antennas, num_pols=1):
        self._capsule = _bda_utils.bda_create(num_antennas, num_pols)

    def set_compression(self, max_fact, fov_deg, wavelength_m, max_avg_time_s):
        return _bda_utils.bda_set_compression(self._capsule, max_fact, fov_deg,
            wavelength_m, max_avg_time_s)

    def set_delta_t(self, value_s):
        _bda_utils.bda_set_delta_t(self._capsule, value_s)

    def set_num_times(self, value):
        _bda_utils.bda_set_num_times(self._capsule, value)

    def set_initial_coords(self, uu, vv, ww):
        _bda_utils.bda_set_initial_coords(self._capsule, uu, vv, ww)

    def add_data(self, time_index, vis, uu_next, vv_next, ww_next):
        _bda_utils.bda_add_data(self._capsule, time_index, vis,
            uu_next, vv_next, ww_next)

    def finalise(self):
        return _bda_utils.bda_finalise(self._capsule)


def apply_gains(vis_amp, gains):
    """
    Apply agains to visibility amplitudes.

    Args:
        vis_amp (array_like):
        gains (array_like):

    returns:
        array of visibilities with gains applied.

    """
    #FIXME(BM) mixed types?
    if vis_amp.dtype == 'c8' and gains.dtype == 'c8':
        return _bda_utils.apply_gains_2(vis_amp, gains)
    else:
        return _bda_utils.apply_gains(vis_amp, gains)


def vis_list_to_matrix(vis_list, num_antennas):
    """
    """
    if vis_list.dtype  == 'c8':
        return _bda_utils.vis_list_to_matrix_2(vis_list, num_antennas)
    else:
        return _bda_utils.vis_list_to_matrix(vis_list, num_antennas)
