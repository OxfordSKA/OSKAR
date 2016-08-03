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

import math
import numpy
import _telescope_lib

class Telescope(object):
    """This class provides a Python interface to an OSKAR telescope model.
    """

    def __init__(self, precision='double', settings_path=None):
        """Creates a handle to an OSKAR telescope model.

        Args:
            precision (str): Either 'double' or 'single' to specify
                the numerical precision of the data.
            settings_path (str): Path to an OSKAR settings file.
        """
        if settings_path is not None:
            self._capsule = _telescope_lib.set_up(settings_path)
        else:
            self._capsule = _telescope_lib.create(precision)


    def load(self, dir_name):
        """Loads an OSKAR telescope model directory.

        Note that some telescope model meta-data must have been set
        prior to calling this method, as it affects how data are loaded.

        Specifically, the following methods must be called BEFORE this one:
            set_position()
            set_pol_mode()
            set_enable_numerical_patterns()

        Args:
            dir_name (str): Path to telescope model directory to load.
        """
        _telescope_lib.load(self._capsule, dir_name)


    def set_allow_station_beam_duplication(self, value):
        """Sets whether station beams will be copied if stations are identical.

        Args:
            value (int):
                If true, station beams will be copied if stations are identical.
        """
        _telescope_lib.set_allow_station_beam_duplication(self._capsule, value)


    def set_channel_bandwidth(self, channel_bandwidth_hz):
        """Sets the value used to simulate bandwidth smearing.

        Args:
            channel_bandwidth_hz (float): The channel bandwidth, in Hz.
        """
        _telescope_lib.set_channel_bandwidth(self._capsule,
            channel_bandwidth_hz)


    def set_enable_noise(self, value, seed=1):
        """Sets whether thermal noise is enabled.

        Args:
            value (int): If true, thermal noise will be added to visibilities.
            seed (int): Random number generator seed.
        """
        _telescope_lib.set_enable_noise(self._capsule, value, seed)


    def set_enable_numerical_patterns(self, value):
        """Sets whether numerical element patterns are enabled.

        Args:
            value (int): If true, numerical element patterns will be loaded.
        """
        _telescope_lib.set_enable_numerical_patterns(self._capsule, value)


    def set_gaussian_station_beam_width(self, fwhm_deg, ref_freq_hz):
        """Sets the parameters used for stations with Gaussian beams.

        Args:
            fwhm_deg (float):
                The Gaussian FWHM value, in degrees.
            ref_freq_hz (float):
                The reference frequency at which the FWHM applies, in Hz.
        """
        _telescope_lib.set_gaussian_station_beam_width(self._capsule,
            fwhm_deg, ref_freq_hz)


    def set_noise_freq(self, start_hz, inc_hz=0.0, num_channels=1):
        """Sets the frequencies at which noise is defined.

        Args:
            start_hz (float):             Start frequency, in Hz.
            inc_hz (Optional[float]):     Frequency increment, in Hz. Default 0.
            num_channels (Optioanl[int]): Number of channels. Default 1.
        """
        _telescope_lib.set_noise_freq(self._capsule,
            start_hz, inc_hz, num_channels)


    def set_noise_rms(self, start, end=None):
        """Sets the noise RMS range.

        Call this only after set_noise_freq().

        Args:
            start (float): Value at first frequency, in Jy.
            end (Optional[float]): Value at last frequency, in Jy.
        """
        if end == None:
            end = start
        _telescope_lib.set_noise_rms(self._capsule, start, end)


    def set_phase_centre(self, ra_deg, dec_deg):
        """Sets the telescope phase centre coordinates.

        Args:
            ra_deg (float): Right Ascension, in degrees.
            dec_deg (float): Declination, in degrees.
        """
        _telescope_lib.set_phase_centre(self._capsule,
            math.radians(ra_deg), math.radians(dec_deg))


    def set_pol_mode(self, mode):
        """Sets the polarisation mode of the telescope.

        Args:
            mode (str): Either 'Scalar' or 'Full'.
        """
        _telescope_lib.set_pol_mode(self._capsule, mode)


    def set_position(self, longitude_deg, latitude_deg, altitude_m=0.0):
        """Sets the reference position of the telescope.

        The position of the telescope must be set before calling 'load'.

        Args:
            longitude_deg (float): Array centre longitude, in degrees.
            latitude_deg (float):  Array centre latitude, in degrees.
            altitude_m (float):    Array centre altitude, in metres.
        """
        _telescope_lib.set_position(self._capsule, math.radians(longitude_deg),
            math.radians(latitude_deg), altitude_m)


    def set_station_coords_enu(self, longitude_deg, latitude_deg, altitude_m,
            x, y, z=None, x_err=None, y_err=None, z_err=None):
        """Sets station coordinates in the East-North-Up (ENU) horizon system.

        Args:
            longitude_deg (float): Array centre longitude, in degrees.
            latitude_deg (float):  Array centre latitude, in degrees.
            altitude_m (float):    Array centre altitude, in metres.
            x (float, array-like):
                Station x coordinates, to East, in metres.
            y (float, array-like):
                Station y coordinates, to North, in metres.
            z (Optional[float, array-like]):
                Station z coordinates, to Up, in metres.
            x_err (Optional[float, array-like]):
                Station x coordinate error, in metres.
            y_err (Optional[float, array-like]):
                Station y coordinate error, in metres.
            z_err (Optional[float, array-like]):
                Station z coordinate error, in metres.
        """
        if z is None:
            z = numpy.zeros_like(x)
        if x_err is None:
            x_err = numpy.zeros_like(x)
        if y_err is None:
            y_err = numpy.zeros_like(x)
        if z_err is None:
            z_err = numpy.zeros_like(x)
        _telescope_lib.set_station_coords_enu(self._capsule,
            math.radians(longitude_deg), math.radians(latitude_deg),
            altitude_m, x, y, z, x_err, y_err, z_err)


    def set_station_coords_ecef(self, longitude_deg, latitude_deg, altitude_m,
            x, y, z, x_err=None, y_err=None, z_err=None):
        """Sets station coordinates in the Earth-centred-Earth-fixed (ECEF) system.

        Args:
            longitude_deg (float): Array centre longitude, in degrees.
            latitude_deg (float):  Array centre latitude, in degrees.
            altitude_m (float):    Array centre altitude, in metres.
            x (float, array-like):
                Station x coordinates, in metres.
            y (float, array-like):
                Station y coordinates, in metres.
            z (float, array-like):
                Station z coordinates, in metres.
            x_err (Optional[float, array-like]):
                Station x coordinate error, in metres.
            y_err (Optional[float, array-like]):
                Station y coordinate error, in metres.
            z_err (Optional[float, array-like]):
                Station z coordinate error, in metres.
        """
        if x_err is None:
            x_err = numpy.zeros_like(x)
        if y_err is None:
            y_err = numpy.zeros_like(x)
        if z_err is None:
            z_err = numpy.zeros_like(x)
        _telescope_lib.set_station_coords_ecef(self._capsule,
            math.radians(longitude_deg), math.radians(latitude_deg),
            altitude_m, x, y, z, x_err, y_err, z_err)


    def set_station_coords_wgs84(self, longitude_deg, latitude_deg, altitude_m,
            station_longitudes_deg, station_latitudes_deg, 
            station_altitudes_m=None):
        """Sets station coordinates in the WGS84 system.

        Args:
            longitude_deg (float): Array centre longitude, in degrees.
            latitude_deg (float):  Array centre latitude, in degrees.
            altitude_m (float):    Array centre altitude, in metres.
            station_longitudes_deg (float, array-like):
                Station longitudes, in degrees.
            station_latitudes_deg (float, array-like):
                Station latitudes, in degrees.
            station_altitudes_m (Optional[float, array-like]):
                Station altitudes, in metres.
        """
        if station_altitudes_m is None:
            station_altitudes_m = numpy.zeros_like(station_longitudes_deg)
        _telescope_lib.set_station_coords_wgs84(self._capsule,
            math.radians(longitude_deg), math.radians(latitude_deg),
            altitude_m, station_longitudes_deg, station_latitudes_deg,
            station_altitudes_m)


    def set_station_type(self, type_string):
        """Sets the type of stations within the telescope model.

        Args:
            type_string (str):
                Station type, either "Array", "Gaussian" or "Isotropic".
                Only the first letter is checked.
        """
        _telescope_lib.set_station_type(self._capsule, type_string)


    def set_time_average(self, time_average_sec):
        """Sets the value used to simulate time smearing.

        Args:
            time_average_sec (float): The time averaging interval, in seconds.
        """
        _telescope_lib.set_time_average(self._capsule, time_average_sec)


    def set_uv_filter(self, uv_filter_min, uv_filter_max, uv_filter_units):
        """Sets the baseline lengths on which visibilities will be evaluated.

        Args:
            uv_filter_min (float): Minimum value for UV filter.
            uv_filter_max (float): Maximum value for UV filter.
            uv_filter_units (str): Units of filter ('Metres' or 'Wavelengths').
        """
        _telescope_lib.set_uv_filter(self._capsule, uv_filter_min,
            uv_filter_max, uv_filter_units)
