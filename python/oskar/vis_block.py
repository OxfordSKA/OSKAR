#
#  This file is part of OSKAR.
#
# Copyright (c) 2016, The University of Oxford
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

from __future__ import absolute_import, division
try:
    from . import _vis_block_lib
except ImportError:
    _vis_block_lib = None


class VisBlock(object):
    """This class provides a Python interface to an OSKAR visibility block."""

    def __init__(self):
        """Constructs a handle to a visibility block."""
        if _vis_block_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = 0

    def auto_correlations(self):
        """Returns an array reference to the auto correlations in the block."""
        return _vis_block_lib.auto_correlations(self._capsule)

    def baseline_uu_metres(self):
        """Returns an array reference to the block baseline uu coordinates."""
        return _vis_block_lib.baseline_uu_metres(self._capsule)

    def baseline_vv_metres(self):
        """Returns an array reference to the block baseline vv coordinates."""
        return _vis_block_lib.baseline_vv_metres(self._capsule)

    def baseline_ww_metres(self):
        """Returns an array reference to the block baseline ww coordinates."""
        return _vis_block_lib.baseline_ww_metres(self._capsule)

    def cross_correlations(self):
        """Returns an array reference to the cross correlations in the block."""
        return _vis_block_lib.cross_correlations(self._capsule)

    def num_baselines(self):
        """Returns the number of baselines in the block."""
        return _vis_block_lib.num_baselines(self._capsule)

    def num_channels(self):
        """Returns the number of frequency channels in the block."""
        return _vis_block_lib.num_channels(self._capsule)

    def num_pols(self):
        """Returns the number of polarisations in the block."""
        return _vis_block_lib.num_pols(self._capsule)

    def num_stations(self):
        """Returns the number of stations in the block."""
        return _vis_block_lib.num_stations(self._capsule)

    def num_times(self):
        """Returns the number of time samples in the block."""
        return _vis_block_lib.num_times(self._capsule)

    def start_channel_index(self):
        """Returns the start channel index of the block."""
        return _vis_block_lib.start_channel_index(self._capsule)

    def start_time_index(self):
        """Returns the start time index of the block."""
        return _vis_block_lib.start_time_index(self._capsule)
