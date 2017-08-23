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

"""Interfaces to the OSKAR visibility header."""

from __future__ import absolute_import, division, print_function
from oskar.binary import Binary
try:
    from . import _vis_header_lib
except ImportError as e:
    print("Import error: " + str(e))
    _vis_header_lib = None


class VisHeader(object):
    """This class provides a Python interface to an OSKAR visibility header."""

    def __init__(self):
        """Constructs a handle to a visibility header."""
        if _vis_header_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = None

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            raise RuntimeError(
                "Call Interferometer.vis_header() for the visibility header.")

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _vis_header_lib.capsule_name(new_capsule) == 'oskar_VisHeader':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_VisHeader.")

    def get_amp_type(self):
        """Returns the OSKAR data type of the visibility amplitude array."""
        self.capsule_ensure()
        return _vis_header_lib.amp_type(self._capsule)

    def get_channel_bandwidth_hz(self):
        """Returns the width of each frequency channel, in Hz."""
        self.capsule_ensure()
        return _vis_header_lib.channel_bandwidth_hz(self._capsule)

    def get_coord_precision(self):
        """Returns the OSKAR data type of the baseline coordinate arrays."""
        self.capsule_ensure()
        return _vis_header_lib.coord_precision(self._capsule)

    def get_freq_inc_hz(self):
        """Returns the frequency channel increment, in Hz."""
        self.capsule_ensure()
        return _vis_header_lib.freq_inc_hz(self._capsule)

    def get_freq_start_hz(self):
        """Returns the frequency of the first channel, in Hz."""
        self.capsule_ensure()
        return _vis_header_lib.freq_start_hz(self._capsule)

    def get_max_channels_per_block(self):
        """Returns the maximum number of channels per visibility block."""
        self.capsule_ensure()
        return _vis_header_lib.max_channels_per_block(self._capsule)

    def get_max_times_per_block(self):
        """Returns the maximum number of time samples per visibility block."""
        self.capsule_ensure()
        return _vis_header_lib.max_times_per_block(self._capsule)

    def get_num_channels_total(self):
        """Returns the total number of frequency channels."""
        self.capsule_ensure()
        return _vis_header_lib.num_channels_total(self._capsule)

    def get_num_stations(self):
        """Returns the number of stations."""
        self.capsule_ensure()
        return _vis_header_lib.num_stations(self._capsule)

    def get_num_tags_per_block(self):
        """Returns the number of binary data tags per visibility block."""
        self.capsule_ensure()
        return _vis_header_lib.num_tags_per_block(self._capsule)

    def get_num_times_total(self):
        """Returns the total number of time samples."""
        self.capsule_ensure()
        return _vis_header_lib.num_times_total(self._capsule)

    def get_phase_centre_ra_deg(self):
        """Returns the phase centre Right Ascension, in degrees."""
        self.capsule_ensure()
        return _vis_header_lib.phase_centre_ra_deg(self._capsule)

    def get_phase_centre_dec_deg(self):
        """Returns the phase centre Declination, in degrees."""
        self.capsule_ensure()
        return _vis_header_lib.phase_centre_dec_deg(self._capsule)

    def get_time_start_mjd_utc(self):
        """Returns the start time, as MJD(UTC)."""
        self.capsule_ensure()
        return _vis_header_lib.time_start_mjd_utc(self._capsule)

    def get_time_inc_sec(self):
        """Returns the time increment, in seconds."""
        self.capsule_ensure()
        return _vis_header_lib.time_inc_sec(self._capsule)

    def get_time_average_sec(self):
        """Returns the time averaging period, in seconds."""
        self.capsule_ensure()
        return _vis_header_lib.time_average_sec(self._capsule)

    # Properties
    amp_type = property(get_amp_type)
    capsule = property(capsule_get, capsule_set)
    channel_bandwidth_hz = property(get_channel_bandwidth_hz)
    coord_precision = property(get_coord_precision)
    freq_inc_hz = property(get_freq_inc_hz)
    freq_start_hz = property(get_freq_start_hz)
    max_channels_per_block = property(get_max_channels_per_block)
    max_times_per_block = property(get_max_times_per_block)
    num_channels_total = property(get_num_channels_total)
    num_stations = property(get_num_stations)
    num_tags_per_block = property(get_num_tags_per_block)
    num_times_total = property(get_num_times_total)
    phase_centre_ra_deg = property(get_phase_centre_ra_deg)
    phase_centre_dec_deg = property(get_phase_centre_dec_deg)
    time_start_mjd_utc = property(get_time_start_mjd_utc)
    time_inc_sec = property(get_time_inc_sec)
    time_average_sec = property(get_time_average_sec)

    @classmethod
    def read(cls, binary_file):
        """Reads a visibility header from an OSKAR binary file and returns it.

        Args:
            binary_file (str or oskar.Binary):
                Path or handle to an OSKAR binary file.

        Returns:
            tuple: A two-element tuple containing the visibility header and
            a handle to the OSKAR binary file, opened for reading.
        """
        if _vis_header_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = VisHeader()
        if isinstance(binary_file, Binary):
            t.capsule = _vis_header_lib.read_header(binary_file.capsule)
            return (t, binary_file)
        else:
            b = Binary(binary_file, 'r')
            t.capsule = _vis_header_lib.read_header(b.capsule)
            return (t, b)
