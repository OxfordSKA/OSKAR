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

import math
import _imager_lib

class Imager(object):
    """This class provides a Python interface to the OSKAR imager."""

    def __init__(self, precision="double"):
        """Creates a handle to an OSKAR imager.

        Args:
            type (str): Either 'double' or 'single' to specify 
                the numerical precision of the images.
        """
        self._capsule, _ = _imager_lib.create(precision)


    def check_init(self):
        """Initialises the imager algorithm if it has not already been done.

        All imager options and data must have been set appropriately 
        before calling this function.
        """
        _imager_lib.check_init(self._capsule)


    def finalise(self, image=None):
        """Finalises the image or images and writes them to file.

        Args:
            image (Optional[float, array like]): 
                If given, the output image is returned in this array.
        """
        _imager_lib.finalise(self._capsule, image)


    def finalise_plane(self, plane, plane_norm):
        """Finalises an image plane.

        This is a low-level function that is used to finalise 
        gridded visibilities when used in conjunction with update_plane().

        The image can be obtained by taking the real part of the plane after 
        this function returns.

        Args:
            plane (complex float, array like): 
                On input, the plane to finalise; on output, the image plane.
            plane_norm (float): Plane normalisation to apply.
        """
        _imager_lib.finalise_plane(self._capsule, plane, plane_norm)


    def plane_size(self):
        """Returns the required plane size.

        This may be different to the image size, for example if using 
        W-projection. It will only be valid after a call to check_init().

        Returns:
            int: Plane side length.
        """
        return _imager_lib.plane_size(self._capsule)


    def reset_cache(self):
        """Low-level function to reset the imager's internal memory.

        This is used to clear any data added using update().
        """
        _imager_lib.reset_cache(self._capsule)


    def run(self, filename):
        """Runs the imager on a visibility file.

        Args:
            filename (str): 
                Path to input Measurement Set or OSKAR visibility file.
        """
        _imager_lib.run(self._capsule, filename)


    def set_algorithm(self, algorithm_type):
        """Sets the algorithm used by the imager.

        Args:
            type (str): Either 'FFT', 'DFT 2D', 'DFT 3D' or 'W-projection'.
        """
        _imager_lib.set_algorithm(self._capsule, algorithm_type)


    def set_default_direction(self):
        """Clears any direction override."""
        _imager_lib.set_default_direction(self._capsule)


    def set_direction(self, ra_deg, dec_deg):
        """Sets the image centre different to the observation phase centre.

        Args:
            ra_deg (float): The new image Right Ascension, in degrees.
            dec_deg (float): The new image Declination, in degrees.
        """
        _imager_lib.set_direction(self._capsule, ra_deg, dec_deg)


    def set_channel_range(self, start, end, snapshots):
        """Sets the visibility channel range used by the imager.

        Args:
            start (int): Start channel index.
            end (int):   End channel index (-1 for all channels).
            snapshots (boolean): If true, image each channel separately; 
                if false, use frequency synthesis.
        """
        _imager_lib.set_channel_range(self._capsule, start, end, snapshots)


    def set_grid_kernel(self, kernel_type, support, oversample):
        """Sets the convolution kernel used for gridding visibilities.

        Args:
            type (str): Type of convolution kernel; 
                either 'Spheroidal' or 'Pillbox'.
            support (int): Support size of kernel. 
                The kernel width is 2 * support + 1.
            oversample (int): Oversample factor used for look-up table.
        """
        _imager_lib.set_grid_kernel(self._capsule, kernel_type, \
            support, oversample)


    def set_image_type(self, image_type):
        """Sets the image (polarisation) type.

        Args:
            type (str): Either 'STOKES', 'I', 'Q', 'U', 'V', 
                'LINEAR', 'XX', 'XY', 'YX', 'YY' or 'PSF'.
        """
        _imager_lib.set_image_type(self._capsule, image_type)


    def set_fft_on_gpu(self, value):
        """Sets whether to use the GPU for FFTs.

        Args:
            value (boolean): If true, use the GPU for FFTs.
        """
        _imager_lib.set_fft_on_gpu(self._capsule, value)


    def set_fov(self, fov_deg):
        """Sets the field of view to image.

        Args:
            fov_deg (float): Field of view, in degrees.
        """
        _imager_lib.set_fov(self._capsule, fov_deg)


    def set_ms_column(self, column):
        """Sets the data column to use from a Measurement Set.

        Args:
            column (str): Name of the column to use.
        """
        _imager_lib.set_ms_column(self._capsule, column)


    def set_output_root(self, filename):
        """Sets the root path of output images.

        Args:
            filename (str): Root path.
        """
        _imager_lib.set_output_root(self._capsule, filename)


    def set_size(self, size):
        """Sets image side length.

        Args:
            size (int): Image side length in pixels.
        """
        _imager_lib.set_size(self._capsule, size)


    def set_time_range(self, start, end, snapshots):
        """Sets the visibility time range used by the imager.

        Args:
            start (int): Start time index.
            end (int):   End time index (-1 for all time).
            snapshots (boolean): If true, image each time slice separately; 
                if false, use time synthesis.
        """
        _imager_lib.set_time_range(self._capsule, start, end, snapshots)


    def set_vis_frequency(self, ref_hz, inc_hz, num_channels):
        """Sets the visibility start frequency.

        Args:
            ref_hz (float): Frequency of index 0, in Hz.
            inc_hz (float): Frequency increment, in Hz.
            num_channels (int): Number of channels in visibility data.
        """
        _imager_lib.set_vis_frequency(self._capsule, \
            ref_hz, inc_hz, num_channels)


    def set_vis_phase_centre(self, ra_deg, dec_deg):
        """Sets the coordinates of the visibility phase centre.

        Args:
            ra_deg (float): Right Ascension of phase centre, in degrees.
            dec_deg (float): Declination of phase centre, in degrees.
        """
        _imager_lib.set_vis_phase_centre(self._capsule, ra_deg, dec_deg)


    def set_vis_time(self, ref_mjd_utc, inc_sec, num_times):
        """Sets the visibility start time.

        Args:
            ref_mjd_utc (float): Time of index 0, as MJD(UTC).
            inc_sec (float): Time increment, in seconds.
            num_times (int): Number of time steps in visibility data.
        """
        _imager_lib.set_vis_time(self._capsule, ref_mjd_utc, inc_sec, num_times)


    def set_w_planes(self, num_planes):
        """Sets the number of W-planes to use, if using W-projection.

        A number less than or equal to zero means 'automatic'.

        Args:
            num_planes (int): Number of W-planes to use.
        """
        _imager_lib.set_w_planes(self._capsule, num_planes)


    def set_w_range(self, w_min, w_max, w_rms):
        """Sets the range of W values, if using W-projection.

        Args:
            w_min (float): Minimum value of w, in wavelengths.
            w_max (float): Maximum value of w, in wavelengths.
            w_rms (float): RMS value of w, in wavelengths.
        """
        _imager_lib.set_w_range(self._capsule, w_min, w_max, w_rms)


    def update(self, num_baselines, uu, vv, ww, amps, weight, 
        num_pols = 1, start_time = 0, end_time = 0, 
        start_channel = 0, end_channel = 0):
        """Runs imager for supplied visibilities, applying optional selection.

        The visibility amplitude data dimension order must be:
        (slowest) time, channel, baseline, polarisation (fastest).

        Call finalise() to finalise the images after calling this function.

        Args:
            num_baselines (int):
                Number of baselines in the visibility block.
            uu (float, array like, shape (n,)):
                Time-baseline ordered uu coordinates, in metres.
            vv (float, array like, shape (n,)):
                Time-baseline ordered vv coordinates, in metres.
            ww (float, array like, shape (n,)):
                Time-baseline ordered ww coordinates, in metres.
            amp (complex float, array like, shape (n,)):
                Baseline visibility amplitudes.
            weight (float, array like, shape (n,)):
                Visibility weights.
            num_pols (Optional[int]):
               Number of polarisations in the visibility block. Default 1.
            start_time (Optional[int]):
               Start time index of the visibility block. Default 0.
            end_time (Optional[int]):
               End time index of the visibility block. Default 0.
            start_chan (Optional[int]):
               Start channel index of the visibility block. Default 0.
            end_chan (Optional[int]):
               End channel index of the visibility block. Default 0.
        """
        _imager_lib.update(self._capsule, num_baselines, uu, vv, ww, 
            amps, weight, num_pols, start_time, end_time, 
            start_channel, end_channel)


    def update_plane(self, uu, vv, ww, amps, weight, plane, plane_norm):
        """Updates the supplied plane with the supplied visibilities.

        This is a low-level function that can be used to generate 
        gridded visibilities if required.

        Call finalise_plane() to finalise the image after calling this function.

        Args:
            uu (float, array like, shape (n,)):
                Baseline uu coordinates, in wavelengths.
            vv (float, array like, shape (n,)):
                Baseline vv coordinates, in wavelengths.
            ww (float, array like, shape (n,)):
                Baseline ww coordinates, in wavelengths.
            amps (complex float, array like, shape (n,)):
                Baseline visibility amplitudes.
            weight (float, array like, shape (n,)):
                Visibility weights.
            plane (float, array like):
                Plane to update.
            plane_norm (float):
                Current plane normalisation.

        Returns:
            float: Updated plane normalisation.
        """
        return _imager_lib.update_plane(self._capsule, uu, vv, ww, 
            amps, weight, plane, plane_norm)


    @staticmethod
    def make_image(uu, vv, ww, amps, weight, fov_deg, size):
        """Makes an image from visibility data.

        Args:
            uu (float, array like, shape (n,)):
                Baseline uu coordinates, in wavelengths.
            vv (float, array like, shape (n,)):
                Baseline vv coordinates, in wavelengths.
            ww (float, array like, shape (n,)):
                Baseline ww coordinates, in wavelengths.
            amps (complex float, array like, shape (n,)):
                Baseline visibility amplitudes.
            weight (float, array like, shape (n,)):
                Visibility weights.
            fov_deg (float): Image field of view, in degrees.
            size (int):      Image size along one dimension, in pixels.

        Returns:
            array: Image as a 2D numpy array. Data are ordered as in FITS image.
        """
        return _imager_lib.make_image(uu, vv, ww, amps, weight, fov_deg, size)


    @staticmethod
    def fov_to_cellsize(fov_rad, size):
        """Convert image FoV and size along one dimension in pixels to cellsize.

        Args:
            fov_rad (float): Image field-of-view, in radians.
            size (int):      Image size in one dimension in pixels.

        Returns:
            float: Image cellsize, in radians.
        """
        rmax = math.sin(0.5 * fov_rad)
        inc = 2.0 * rmax / size
        return math.asin(inc)


    @staticmethod
    def fov_to_uv_cellsize(fov_rad, size):
        """Convert image FoV and size along one dimension in pixels to cellsize.

        Args:
            fov_rad (float): Image field-of-view, in radians.
            size (int):      Image size in one dimension in pixels.

        Returns:
            float: UV cellsize, in wavelengths.
        """
        return 1.0 / (size * Imager.fov_to_cellsize(fov_rad, size))

