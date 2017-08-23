# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2017, The University of Oxford
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

"""Interfaces to the OSKAR visibility block."""

from __future__ import absolute_import, division, print_function
try:
    from . import _vis_block_lib
except ImportError as e:
    print("Import error: " + str(e))
    _vis_block_lib = None


class VisBlock(object):
    """This class provides a Python interface to an OSKAR visibility block."""

    def __init__(self):
        """Constructs a handle to a visibility block."""
        if _vis_block_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = None

    def auto_correlations(self):
        """Returns an array reference to the auto correlations."""
        self.capsule_ensure()
        return _vis_block_lib.auto_correlations(self._capsule)

    def baseline_uu_metres(self):
        """Returns an array reference to the block baseline uu coordinates."""
        self.capsule_ensure()
        return _vis_block_lib.baseline_uu_metres(self._capsule)

    def baseline_vv_metres(self):
        """Returns an array reference to the block baseline vv coordinates."""
        self.capsule_ensure()
        return _vis_block_lib.baseline_vv_metres(self._capsule)

    def baseline_ww_metres(self):
        """Returns an array reference to the block baseline ww coordinates."""
        self.capsule_ensure()
        return _vis_block_lib.baseline_ww_metres(self._capsule)

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            raise RuntimeError(
                "Creation of VisBlock in Python is not yet supported.")

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _vis_block_lib.capsule_name(new_capsule) == 'oskar_VisBlock':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_VisBlock.")

    def cross_correlations(self):
        """Returns an array reference to the cross correlations."""
        self.capsule_ensure()
        return _vis_block_lib.cross_correlations(self._capsule)

    def get_num_baselines(self):
        """Returns the number of baselines."""
        self.capsule_ensure()
        return _vis_block_lib.num_baselines(self._capsule)

    def get_num_channels(self):
        """Returns the number of frequency channels."""
        self.capsule_ensure()
        return _vis_block_lib.num_channels(self._capsule)

    def get_num_pols(self):
        """Returns the number of polarisations."""
        self.capsule_ensure()
        return _vis_block_lib.num_pols(self._capsule)

    def get_num_stations(self):
        """Returns the number of stations."""
        self.capsule_ensure()
        return _vis_block_lib.num_stations(self._capsule)

    def get_num_times(self):
        """Returns the number of time samples."""
        self.capsule_ensure()
        return _vis_block_lib.num_times(self._capsule)

    def get_start_channel_index(self):
        """Returns the start channel index."""
        self.capsule_ensure()
        return _vis_block_lib.start_channel_index(self._capsule)

    def get_start_time_index(self):
        """Returns the start time index."""
        self.capsule_ensure()
        return _vis_block_lib.start_time_index(self._capsule)

    # Properties
    capsule = property(capsule_get, capsule_set)
    num_baselines = property(get_num_baselines)
    num_channels = property(get_num_channels)
    num_pols = property(get_num_pols)
    num_stations = property(get_num_stations)
    num_times = property(get_num_times)
    start_channel_index = property(get_start_channel_index)
    start_time_index = property(get_start_time_index)
