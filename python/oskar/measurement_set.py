# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2017, The University of Oxford
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

"""Interfaces to a Measurement Set."""

from __future__ import absolute_import, division, print_function
try:
    from . import _measurement_set_lib
except ImportError:
    _measurement_set_lib = None


class MeasurementSet(object):
    """This class provides a Python interface to a CASA Measurement Set.

    The :class:`oskar.MeasurementSet` class can be used to read data from
    and write data to a CASA Measurement Set.
    The `casacore library <https://github.com/casacore/casacore>`_ is used
    for this interface, so it must be available when OSKAR is compiled for
    this to work.

    The Measurement Set format is extremely flexible, and casacore provides a
    complete and comprehensive interface to use it fully.
    However, in many cases a simpler interface is sufficient and may be
    easier to use. This class can be used to read and write Measurement Sets
    containing only a single spectral window (but potentially multiple channels)
    and a single field - this includes all Measurement Sets written by OSKAR.

    Create or open a Measurement Set using the
    :meth:`create() <oskar.MeasurementSet.create>` and
    :meth:`open() <oskar.MeasurementSet.open>` methods, respectively.
    Both return a new :class:`oskar.MeasurementSet` object.

    Once opened, to read data from the Measurement Set:

    - the :meth:`read_column() <oskar.MeasurementSet.read_column>` method
      can be used to read any named column of data from the main table;
    - the :meth:`read_vis()  <oskar.MeasurementSet.read_vis>` convenience
      method can be used to read visibility data from one of the data columns
      in the main table;
    - the :meth:`read_coords()  <oskar.MeasurementSet.read_coords>` convenience
      method can be used to read baseline (u,v,w) coordinates from the
      main table.

    Once created, to write data to a Measurement Set:

    - the :meth:`write_vis()  <oskar.MeasurementSet.write_vis>`
      method can be used to write visibility data to the DATA column;
    - the :meth:`write_coords()  <oskar.MeasurementSet.write_coords>`
      method can be used to write time and baseline coordinates to the
      main table.
      (The baseline order is implicit, so the antenna indices will be
      written automatically.)

    Examples:

        To write a Measurement Set from Python:

        .. code-block:: python

            import numpy

            filename = "test1.ms"

            # Define data dimensions.
            num_pols = 4
            num_channels = 2
            num_stations = 3
            num_times = 4
            num_baselines = num_stations * (num_stations - 1) // 2
            ref_freq_hz = 100e6
            freq_inc_hz = 100e3
            exposure_sec = 1.0
            interval_sec = 1.0

            # Data to write are stored as numpy arrays.
            uu = numpy.zeros([num_baselines])
            vv = numpy.zeros_like(uu)
            ww = numpy.zeros_like(uu)
            vis = numpy.zeros([num_times, num_channels,
                               num_baselines, num_pols], dtype='c8')

            # Create the empty Measurement Set.
            ms = oskar.MeasurementSet.create(filename, num_stations,
                                             num_channels, num_pols,
                                             ref_freq_hz, freq_inc_hz)

            # Set phase centre.
            ra_rad = numpy.pi / 4
            dec_rad = -numpy.pi / 4
            ms.set_phase_centre(ra_rad, dec_rad)

            # Write data one block at a time.
            for t in range(num_times):
                # Dummy data to write.
                time_stamp = 51544.5 * 86400.0 + t
                uu[:] = 1e0 * t + 1
                vv[:] = 1e1 * t + 2
                ww[:] = 1e2 * t + 3
                for c in range(num_channels):
                    for b in range(num_baselines):
                        vis[t, c, b, :] = (t * 10 + b) + 1j * (c + 1)

                # Write coordinates and visibilities.
                start_row = t * num_baselines
                ms.write_coords(start_row, num_baselines, uu, vv, ww,
                                exposure_sec, interval_sec, time_stamp)
                ms.write_vis(start_row, 0, num_channels, num_baselines, vis[t, ...])

    """

    def __init__(self):
        """The default constructor does nothing. Use create() or open()."""
        if _measurement_set_lib is None:
            raise RuntimeError(
                "OSKAR was compiled without Measurement Set support.")
        self._capsule = None

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _measurement_set_lib.capsule_name(new_capsule) == \
                'oskar_MeasurementSet':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_MeasurementSet.")

    @classmethod
    def create(cls, file_name, num_stations, num_channels, num_pols,
               ref_freq_hz, freq_inc_hz, write_autocorr=False,
               write_crosscorr=True):
        """Creates a new, empty Measurement Set with the given name.

        Args:
            file_name (str):      File name of the new Measurement Set.
            num_stations (int):   The number of antennas/stations.
            num_channels (int):   The number of channels in the band.
            num_pols (int):       The number of polarisations (1, 2 or 4).
            ref_freq_hz (float):
                The frequency at the centre of channel 0, in Hz.
            freq_inc_hz (float):
                The increment between channels, in Hz.
            write_autocorr (Optional[bool]):
                If set, allow for write of auto-correlation data.
            write_crosscorr (Optional[bool]):
                If set, allow for write of cross-correlation data.
        """
        if _measurement_set_lib is None:
            raise RuntimeError(
                "OSKAR was compiled without Measurement Set support.")
        t = MeasurementSet()
        t.capsule = _measurement_set_lib.create(
            file_name, num_stations, num_channels, num_pols,
            ref_freq_hz, freq_inc_hz, write_autocorr, write_crosscorr)
        return t

    def ensure_num_rows(self, num):
        """Ensures the specified number of rows exist in the Measurement Set.

        Note that rows will not be removed if the total number is already
        larger than the value given.

        Args:
            num (int):   Ensure at least this many rows are present.
        """
        _measurement_set_lib.ensure_num_rows(self._capsule, num)

    def get_freq_inc_hz(self):
        """Returns the frequency increment between channels."""
        return _measurement_set_lib.freq_inc_hz(self._capsule)

    def get_freq_start_hz(self):
        """Returns the frequency at the centre of the first channel."""
        return _measurement_set_lib.freq_start_hz(self._capsule)

    def get_num_channels(self):
        """Returns the number of frequency channels in the Measurement Set."""
        return _measurement_set_lib.num_channels(self._capsule)

    def get_num_pols(self):
        """Returns the number of polarisations in the Measurement Set."""
        return _measurement_set_lib.num_pols(self._capsule)

    def get_num_rows(self):
        """Returns the number of rows in the Measurement Set main table."""
        return _measurement_set_lib.num_rows(self._capsule)

    def get_num_stations(self):
        """Returns the number of stations in the Measurement Set."""
        return _measurement_set_lib.num_stations(self._capsule)

    def get_phase_centre_ra_rad(self):
        """Returns the Right Ascension of the phase centre in radians."""
        return _measurement_set_lib.phase_centre_ra_rad(self._capsule)

    def get_phase_centre_dec_rad(self):
        """Returns the Declination of the phase centre in radians."""
        return _measurement_set_lib.phase_centre_dec_rad(self._capsule)

    @classmethod
    def open(cls, file_name, readonly=False):
        """Opens an existing Measurement Set with the given name.

        Args:
            file_name (str):      File name of the Measurement Set to open.
            readonly  (bool):     Open the Measurement Set in read-only mode.
        """
        if _measurement_set_lib is None:
            raise RuntimeError(
                "OSKAR was compiled without Measurement Set support.")
        t = MeasurementSet()
        t.capsule = _measurement_set_lib.open(file_name, readonly)
        return t

    def read_column(self, column, start_row, num_rows):
        """Reads a column of data from the main table.

        Args:
            column (str):           Name of the column to read.
            start_row (int):        The start row index to read (zero-based).
            num_rows (int):         Number of rows to read.

        Returns:
            numpy.array: A numpy array containing the contents of the column.
        """
        return _measurement_set_lib.read_column(
            self._capsule, column, start_row, num_rows)

    def read_coords(self, start_row, num_baselines):
        """Reads baseline coordinate data from the main table.

        This function reads a list of baseline coordinates from
        the main table of the Measurement Set. The coordinates are returned
        as a tuple of numpy arrays (uu, vv, ww).

        Args:
            start_row (int):        The start row index to read (zero-based).
            num_baselines (int):    Number of baselines (rows) to read.

        Returns:
            tuple: (uu, vv, ww) baseline coordinates, in metres.
        """
        return _measurement_set_lib.read_coords(
            self._capsule, start_row, num_baselines)

    def read_vis(self, start_row, start_channel, num_channels, num_baselines,
                 column='DATA'):
        """Reads visibility data from the main table.

        This function reads a block of visibility data from the specified
        column of the main table of the Measurement Set.

        The dimensionality of the returned data block is:
        (num_channels * num_baselines * num_pols),
        with num_pols the fastest varying dimension, then num_baselines,
        and num_channels the slowest.

        Args:
            start_row (int):        The start row index to read (zero-based).
            start_channel (int):    Start channel index to read.
            num_channels (int):     Number of channels to read.
            num_baselines (int):    Number of baselines to read.
            column (Optional[str]): Name of the data column to read.

        Returns:
            array: visibility data block.
        """
        return _measurement_set_lib.read_vis(
            self._capsule, start_row, start_channel,
            num_channels, num_baselines, column)

    def set_phase_centre(self, longitude_rad, latitude_rad):
        """Sets the phase centre.

        Args:
            longitude_rad (float):  Right Ascension of phase centre.
            latitude_rad (float):   Declination of phase centre.
        """
        return _measurement_set_lib.set_phase_centre(
            self._capsule, longitude_rad, latitude_rad)

    def write_coords(self, start_row, num_baselines, uu, vv, ww,
                     exposure_sec, interval_sec, time_stamp):
        """Writes baseline coordinate data to the main table.

        This function writes the supplied list of baseline coordinates to
        the main table of the Measurement Set, extending it if necessary.

        Baseline antenna-pair ordering is implicit:
        a0-a1, a0-a2, a0-a3... a1-a2, a1-a3... a2-a3 etc.
        The supplied number of baselines must be compatible with the number of
        stations in the Measurement Set. Auto-correlations are allowed.

        This function should be called for each time step to write out the
        baseline coordinate data.

        The time stamp is given in units of (MJD) * 86400, i.e. seconds since
        Julian date 2400000.5.

        Args:
            start_row (int):        The start row index to write (zero-based).
            num_baselines (int):    Number of rows to add to the main table.
            uu (float, array-like): Baseline u-coordinates, in metres.
            vv (float, array-like): Baseline v-coordinates, in metres.
            ww (float, array-like): Baseline w-coordinates, in metres.
            exposure_sec (float):   Exposure length per visibility, in seconds.
            interval_sec (float):   Interval length per visibility, in seconds.
            time_stamp (float):     Time stamp of visibility block.
        """
        _measurement_set_lib.write_coords(
            self._capsule, start_row, num_baselines, uu, vv, ww,
            exposure_sec, interval_sec, time_stamp)

    def write_vis(self, start_row, start_channel, num_channels,
                  num_baselines, vis):
        """Writes visibility data to the main table.

        This function writes the given block of visibility data to the
        data column of the Measurement Set, extending it if necessary.

        The dimensionality of the complex vis data block is:
        (num_channels * num_baselines * num_pols),
        with num_pols the fastest varying dimension, then num_baselines,
        and num_channels the slowest.

        Args:
            start_row (int):     The start row index to write (zero-based).
            start_channel (int): Start channel index of the visibility block.
            num_channels (int):  Number of channels in the visibility block.
            num_baselines (int): Number of baselines in the visibility block.
            vis (complex float, array-like): Pointer to visibility block.
        """
        _measurement_set_lib.write_vis(
            self._capsule, start_row, start_channel, num_channels,
            num_baselines, vis)

    # Properties
    capsule = property(capsule_get, capsule_set)
    freq_inc_hz = property(get_freq_inc_hz)
    freq_start_hz = property(get_freq_start_hz)
    num_channels = property(get_num_channels)
    num_pols = property(get_num_pols)
    num_rows = property(get_num_rows)
    num_stations = property(get_num_stations)
    phase_centre_ra_rad = property(get_phase_centre_ra_rad)
    phase_centre_dec_rad = property(get_phase_centre_dec_rad)
