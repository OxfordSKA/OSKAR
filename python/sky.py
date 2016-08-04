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
import _sky_lib

class Sky(object):
    """This class provides a Python interface to an OSKAR sky model."""

    def __init__(self, precision='double'):
        """Creates a handle to an OSKAR sky model.

        Args:
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the data.
        """
        self._capsule = _sky_lib.create(precision)


    def append(self, other):
        """Appends data from another sky model.

        Args:
            other (oskar.Sky): Another sky model.
        """
        _sky_lib.append(self._capsule, other._capsule)


    def append_sources(self, ra_deg, dec_deg, I, Q=None, U=None, V=None, 
            ref_freq_hz=None, spectral_index=None, rotation_measure=None,
            major_axis_arcsec=None, minor_axis_arcsec=None, 
            position_angle_deg=None):
        """Appends source data to an OSKAR sky model from arrays in memory.

        Args:
            ra_deg (float, array-like):
                Source Right Ascension values, in degrees.
            dec_deg (float, array-like):
                Source Declination values, in degrees.
            I (float, array-like):           Source Stokes I fluxes, in Jy.
            Q (Optional[float, array-like]): Source Stokes Q fluxes, in Jy.
            U (Optional[float, array-like]): Source Stokes U fluxes, in Jy.
            V (Optional[float, array-like]): Source Stokes V fluxes, in Jy.
            ref_freq_hz (Optional[float, array-like]):
                Source reference frequency values, in Hz.
            spectral_index (Optional[float, array-like]):
                Source spectral index values.
            rotation_measure (Optional[float, array-like]):
                Source rotation measure values, in rad/m^2.
            major_axis_arcsec (Optional[float, array-like]):
                Source Gaussian major axis values, in arcsec.
            minor_axis_arcsec (Optional[float, array-like]):
                Source Gaussian minor axis values, in arcsec.
            position_angle_deg (Optional[float, array-like]):
                Source Gaussian position angle values, in degrees.
        """
        if Q is None:
            Q = numpy.zeros_like(I)
        if U is None:
            U = numpy.zeros_like(I)
        if V is None:
            V = numpy.zeros_like(I)
        if ref_freq_hz is None:
            ref_freq_hz = numpy.zeros_like(I)
        if spectral_index is None:
            spectral_index = numpy.zeros_like(I)
        if rotation_measure is None:
            rotation_measure = numpy.zeros_like(I)
        if major_axis_arcsec is None:
            major_axis_arcsec = numpy.zeros_like(I)
        if minor_axis_arcsec is None:
            minor_axis_arcsec = numpy.zeros_like(I)
        if position_angle_deg is None:
            position_angle_deg = numpy.zeros_like(I)
        _sky_lib.append_sources(self._capsule, 
            numpy.radians(ra_deg), numpy.radians(dec_deg), I, Q, U, V, 
            ref_freq_hz, spectral_index, rotation_measure, 
            numpy.radians(major_axis_arcsec / 3600.0), 
            numpy.radians(minor_axis_arcsec / 3600.0), 
            numpy.radians(position_angle_deg))


    def append_file(self, filename):
        """Appends data to the sky model from a text file.

        Args:
            filename (str): Name of file to load.
        """
        _sky_lib.append_file(self._capsule, filename)


    def create_copy(self):
        """Creates a copy of the sky model."""
        t = Sky()
        t._capsule = _sky_lib.create_copy(self._capsule)
        return t


    def filter_by_flux(self, min_flux_jy, max_flux_jy):
        """Filters the sky model according to Stokes-I flux.

        Sources with flux values outside the range are deleted.

        Args:
            min_flux_jy (float): Minimum allowed flux, in Jy.
            max_flux_jy (float): Maximum allowed flux, in Jy.
        """
        _sky_lib.filter_by_flux(self._capsule, min_flux_jy, max_flux_jy)


    def filter_by_radius(self, inner_radius_deg, outer_radius_deg, 
            ra0_deg, dec0_deg):
        """Filters the sky model according to source radius from phase centre.

        Sources outside the range are deleted.

        Args:
            inner_radius_deg (float): Minimum allowed radius, in degrees.
            outer_radius_deg (float): Maximum allowed radius, in degrees.
            ra0_deg (float): Right Ascension of phase centre, in degrees.
            dec0_deg (float): Declination of phase centre, in degrees.
        """
        _sky_lib.filter_by_radius(self._capsule, 
            math.radians(inner_radius_deg), math.radians(outer_radius_deg), 
            math.radians(ra0_deg), math.radians(dec0_deg))


    @classmethod
    def from_fits_file(cls, filename, min_peak_fraction=0.0, min_abs_val=0.0,
            default_map_units='K', override_units=False,
            frequency_hz=0.0, spectral_index=-0.7, precision='double'):
        """Loads data from a FITS file and returns it as a new sky model.

        The file can be either a regular FITS image
        or a HEALPix FITS file in RING format.

        Args:
            filename (str): Name of FITS file to load.
            min_peak_fraction (Optional[float]):
                Minimum pixel value loaded, as a fraction of the image peak.
            min_abs_val (Optional[float]):
                Minimum pixel value loaded.
            default_map_units (Optional[str]):
                Default map units, if not found in the file.
                Can be 'Jy/beam', 'Jy/pixel', 'K' or 'mK'.
            override_units (Optional[boolean]):
                If true, override image units with the default.
            frequency_hz (Optional[float]):
                Frequency of the image data in Hz, if not found in the file.
            spectral_index (Optional[float]):
                Spectral index value to give to each pixel.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the data.
        """
        t = Sky()
        t._capsule = _sky_lib.from_fits_file(filename, min_peak_fraction,
            min_abs_val, default_map_units, override_units, frequency_hz,
            spectral_index, precision)
        return t


    @classmethod
    def generate_grid(cls, ra0_deg, dec0_deg, side_length, fov_deg,
            mean_flux_jy=1.0, std_flux_jy=0.0, seed=1, precision='double'):
        """Generates a grid of sources and returns it as a new sky model.

        Args:
            ra0_deg (float):      Right Ascension of grid centre, in degrees.
            dec0_deg (float):     Declination of grid centre, in degrees.
            side_length (int):    Side length of generated grid.
            fov_deg (float):      Grid field-of-view, in degrees.
            mean_flux_jy (float): Mean Stokes-I source flux, in Jy.
            std_flux_jy (float):  Standard deviation Stokes-I source flux, in Jy.
            seed (int):           Random generator seed.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the data.
        """
        t = Sky()
        t._capsule = _sky_lib.generate_grid(ra0_deg, dec0_deg, side_length,
            fov_deg, mean_flux_jy, std_flux_jy, seed, precision)
        return t


    @classmethod
    def generate_random_power_law(cls, num_sources, min_flux_jy, max_flux_jy,
            power_law_index, seed=1, precision='double'):
        """Generates sources scattered randomly over the sphere.

        Args:
            num_sources (int):       The number of sources to generate.
            min_flux_jy (float):     Minimum Stokes-I source flux, in Jy.
            max_flux_jy (float):     Maximum Stokes-I source flux, in Jy.
            power_law_index (float): Power law index/exponent.
            seed (int):              Random generator seed.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the data.
        """
        t = Sky()
        t._capsule = _sky_lib.generate_random_power_law(num_sources,
            min_flux_jy, max_flux_jy, power_law_index, seed, precision)
        return t


    @classmethod
    def load(cls, filename, precision='double'):
        """Loads data from a text file and returns it as a new sky model.

        Args:
            filename (str): Name of file to load.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the data.
        """
        t = Sky()
        t._capsule = _sky_lib.load(filename, precision)
        return t


    def save(self, filename):
        """Saves data to a sky model text file.

        Args:
            filename (str): Name of file to write.
        """
        _sky_lib.save(self._capsule, filename)

