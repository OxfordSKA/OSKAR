# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2017, The University of Oxford
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

"""Interfaces to the OSKAR imager."""

from __future__ import absolute_import, division, print_function
import math
import numpy
try:
    from . import _imager_lib
except ImportError as e:
    print("Import error: " + str(e))
    _imager_lib = None


class Imager(object):
    """This class provides a Python interface to the OSKAR imager."""

    def __init__(self, precision=None, settings=None):
        """Creates an OSKAR imager.

        Args:
            precision (Optional[str]):
                Either 'double' or 'single' to specify the numerical
                precision of the images. Default 'double'.
            settings (Optional[oskar.SettingsTree]):
                Optional settings to use to set up the imager.
        """
        if _imager_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = None
        if precision is not None and settings is not None:
            raise RuntimeError("Specify either precision or all settings.")
        if precision is None:
            precision = 'double'  # Set default.
        if settings is not None:
            img = settings.to_imager()
            self._capsule = img.capsule
        self._precision = precision

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            self._capsule = _imager_lib.create(self._precision)

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _imager_lib.capsule_name(new_capsule) == 'oskar_Imager':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_Imager.")

    def check_init(self):
        """Initialises the imager algorithm if it has not already been done.

        All imager options and data must have been set appropriately
        before calling this function.
        """
        self.capsule_ensure()
        _imager_lib.check_init(self._capsule)

    def finalise(self, return_images=0, return_grids=0):
        """Finalises the image or images.

        Images or grids can be returned in a Python dictionary
        of numpy arrays, if required.
        The image cube can be accessed using the 'images' key, and
        the grid cube can be accessed using the 'grids' key.

        Args:
            return_images (Optional[int]): Number of image planes to return.
            return_grids (Optional[int]): Number of grid planes to return.
        """
        self.capsule_ensure()
        return _imager_lib.finalise(self._capsule, return_images, return_grids)

    def finalise_plane(self, plane, plane_norm):
        """Finalises an image plane.

        This is a low-level function that is used to finalise
        gridded visibilities when used in conjunction with update_plane().

        The image can be obtained by taking the real part of the plane after
        this function returns.

        Args:
            plane (complex float, array-like):
                On input, the plane to finalise; on output, the image plane.
            plane_norm (float): Plane normalisation to apply.
        """
        self.capsule_ensure()
        _imager_lib.finalise_plane(self._capsule, plane, plane_norm)

    def get_algorithm(self):
        """Returns a string describing the imager algorithm.

        Returns:
            str: The imager algorithm.
        """
        self.capsule_ensure()
        return _imager_lib.algorithm(self._capsule)

    def get_cellsize(self):
        """Returns the image cell (pixel) size.

        Returns:
            float: The cell size, in arcsec.
        """
        self.capsule_ensure()
        return _imager_lib.cellsize(self._capsule)

    def get_channel_snapshots(self):
        """Returns the flag specifying whether to image each channel separately.

        Returns:
            boolean: If true, image each channel index separately;
                if false, use frequency synthesis.
        """
        self.capsule_ensure()
        return _imager_lib.channel_snapshots(self._capsule)

    def get_coords_only(self):
        """Returns flag specifying whether imager is in coordinate-only mode.

        Returns:
            boolean: If true, imager is in coordinate-only mode.
        """
        self.capsule_ensure()
        return _imager_lib.coords_only(self._capsule)

    def get_fft_on_gpu(self):
        """Returns flag specifying whether to use the GPU for FFTs.

        Returns:
            boolean: If true, use the GPU for FFTs.
        """
        self.capsule_ensure()
        return _imager_lib.fft_on_gpu(self._capsule)

    def get_fov(self):
        """Returns the image field-of-view, in degrees.

        Returns:
            float: The image field-of-view, in degrees.
        """
        self.capsule_ensure()
        return _imager_lib.fov(self._capsule)

    def get_freq_max_hz(self):
        """Returns the maximum frequency of visibility data to image.

        Returns:
            float: The maximum frequency of visibility data to image, in Hz.
        """
        self.capsule_ensure()
        return _imager_lib.freq_max_hz(self._capsule)

    def get_freq_min_hz(self):
        """Returns the minimum frequency of visibility data to image.

        Returns:
            float: The minimum frequency of visibility data to image, in Hz.
        """
        self.capsule_ensure()
        return _imager_lib.freq_min_hz(self._capsule)

    def get_generate_w_kernels_on_gpu(self):
        """Returns flag specifying whether to use the GPU to generate W-kernels.

        Returns:
            boolean: If true, use the GPU to generate W-kernels.
        """
        self.capsule_ensure()
        return _imager_lib.generate_w_kernels_on_gpu(self._capsule)

    def get_image_size(self):
        """Returns the image side length, in pixels.

        Returns:
            int: The image side length, in pixels.
        """
        self.capsule_ensure()
        return _imager_lib.image_size(self._capsule)

    def get_image_type(self):
        """Returns a string describing the image (polarisation) type.

        Returns:
            str: The image (polarisation) type.
        """
        self.capsule_ensure()
        return _imager_lib.image_type(self._capsule)

    def get_input_file(self):
        """Returns a string containing the input file name.

        Returns:
            str: The input file name.
        """
        self.capsule_ensure()
        return _imager_lib.input_file(self._capsule)

    def get_ms_column(self):
        """Returns a string containing the Measurement Set column to use.

        Returns:
            str: The column name.
        """
        self.capsule_ensure()
        return _imager_lib.ms_column(self._capsule)

    def get_num_w_planes(self):
        """Returns the number of W-planes used.

        Returns:
            int: The number of W-planes used.
        """
        self.capsule_ensure()
        return _imager_lib.num_w_planes(self._capsule)

    def get_output_root(self):
        """Returns a string containing the output root file name.

        Returns:
            str: The output root file name.
        """
        self.capsule_ensure()
        return _imager_lib.output_root(self._capsule)

    def get_plane_size(self):
        """Returns the required plane size.

        This may be different to the image size, for example if using
        W-projection. It will only be valid after a call to check_init().

        Returns:
            int: Plane side length.
        """
        self.capsule_ensure()
        return _imager_lib.plane_size(self._capsule)

    def get_scale_norm_with_num_input_files(self):
        """Returns the option to scale image normalisation by the number of
        input files.

        Returns:
            boolean: The option value (true or false).
        """
        self.capsule_ensure()
        return _imager_lib.scale_norm_with_num_input_files(self._capsule)

    def get_size(self):
        """Returns the image side length, in pixels.

        Returns:
            int: The image side length, in pixels.
        """
        self.capsule_ensure()
        return _imager_lib.size(self._capsule)

    def get_time_max_utc(self):
        """Returns the maximum time of visibility data to include in the image.

        Returns:
            float: The maximum time of visibility data, as MJD(UTC).
        """
        self.capsule_ensure()
        return _imager_lib.time_max_utc(self._capsule)

    def get_time_min_utc(self):
        """Returns the minimum time of visibility data to include in the image.

        Returns:
            float: The minimum time of visibility data, as MJD(UTC).
        """
        self.capsule_ensure()
        return _imager_lib.time_min_utc(self._capsule)

    def get_uv_filter_max(self):
        """Returns the maximum UV baseline length to image, in wavelengths.

        Returns:
            float: Maximum UV baseline length to image, in wavelengths.
        """
        self.capsule_ensure()
        return _imager_lib.uv_filter_max(self._capsule)

    def get_uv_filter_min(self):
        """Returns the minimum UV baseline length to image, in wavelengths.

        Returns:
            float: Minimum UV baseline length to image, in wavelengths.
        """
        self.capsule_ensure()
        return _imager_lib.uv_filter_min(self._capsule)

    def get_weighting(self):
        """Returns a string describing the weighting scheme.

        Returns:
            str: The weighting scheme.
        """
        self.capsule_ensure()
        return _imager_lib.weighting(self._capsule)

    def reset_cache(self):
        """Low-level function to reset the imager's internal memory.

        This is used to clear any data added using update().
        """
        self.capsule_ensure()
        _imager_lib.reset_cache(self._capsule)

    def rotate_coords(self, uu_in, vv_in, ww_in):
        """Rotates baseline coordinates to the new phase centre (if set).

        Prior to calling this method, the new phase centre must be set first
        using set_direction(), and then the original phase centre
        must be set using set_vis_phase_centre().
        Note that the order of these calls is important.

        Args:
            uu_in (numpy.ndarray): Baseline uu coordinates.
            vv_in (numpy.ndarray): Baseline vv coordinates.
            ww_in (numpy.ndarray): Baseline ww coordinates.
        """
        self.capsule_ensure()
        _imager_lib.rotate_coords(self._capsule, uu_in, vv_in, ww_in)

    def rotate_vis(self, uu_in, vv_in, ww_in, vis):
        """Phase-rotates visibility amplitudes to the new phase centre (if set).

        Prior to calling this method, the new phase centre must be set first
        using set_direction(), and then the original phase centre
        must be set using set_vis_phase_centre().
        Note that the order of these calls is important.

        Note that the coordinates (uu_in, vv_in, ww_in) correspond to the
        original phase centre, and must be in wavelengths.

        Args:
            uu_in (float, array-like):
                Original baseline uu coordinates, in wavelengths.
            vv_in (float, array-like):
                Original baseline vv coordinates, in wavelengths.
            ww_in (float, array-like):
                Original baseline ww coordinates, in wavelengths.
            vis (numpy.ndarray):
                Complex visibility amplitudes.
        """
        self.capsule_ensure()
        _imager_lib.rotate_vis(self._capsule, uu_in, vv_in, ww_in, vis)

    def run(self, uu=None, vv=None, ww=None, amps=None, weight=None,
            time_centroid=None, start_channel=0, end_channel=0,
            num_pols=1, return_images=0, return_grids=0):
        """Runs the imager.

        Visibilities will be used either from the input file or
        Measurement Set, if one is set, or from the supplied arrays.
        If using a file, the input filename must be set using set_input_file().
        If using arrays, the visibility meta-data must be set prior to calling
        this method using set_vis_* methods.

        The visibility amplitude data dimension order must be:
        (slowest) time/baseline, channel, polarisation (fastest).
        This order is the same as that stored in a Measurement Set.

        The visibility weight data dimension order must be:
        (slowest) time/baseline, polarisation (fastest).

        If not given, the weights will be treated as all 1.

        Images or grids can be returned in a Python dictionary
        of numpy arrays, if required.
        The image cube can be accessed using the 'images' key, and
        the grid cube can be accessed using the 'grids' key.

        Args:
            uu (float, array-like, shape (n,)):
                Time-baseline ordered uu coordinates, in metres.
            vv (float, array-like, shape (n,)):
                Time-baseline ordered vv coordinates, in metres.
            ww (float, array-like, shape (n,)):
                Time-baseline ordered ww coordinates, in metres.
            amps (complex float, array-like, shape (m,)):
                Baseline visibility amplitudes. Length as described above.
            weight (Optional[float, array-like, shape (p,)]):
                Visibility weights. Length as described above.
            time_centroid (Optional[float, array-like, shape (n,)]):
                Visibility time centroid values, as MJD(UTC) seconds.
            start_channel (Optional[int]):
                Start channel index of the visibility block. Default 0.
            end_channel (Optional[int]):
                End channel index of the visibility block. Default 0.
            num_pols (Optional[int]):
                Number of polarisations in the visibility block. Default 1.
            return_images (Optional[int]): Number of image planes to return.
            return_grids (Optional[int]): Number of grid planes to return.
        """
        self.capsule_ensure()
        if uu is None:
            return _imager_lib.run(self._capsule, return_images, return_grids)
        else:
            self.reset_cache()
            if self.weighting == 'Uniform' or self.algorithm == 'W-projection':
                self.set_coords_only(True)
                self.update(uu, vv, ww, amps, weight, time_centroid,
                            start_channel, end_channel, num_pols)
                self.set_coords_only(False)
            self.update(uu, vv, ww, amps, weight, time_centroid,
                        start_channel, end_channel, num_pols)
            return self.finalise(return_images, return_grids)

    def set(self, **kwargs):
        """Sets multiple properties at once.

        Example: set(fov_deg=2.0, image_size=2048, algorithm='W-projection')
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_algorithm(self, algorithm_type):
        """Sets the algorithm used by the imager.

        Args:
            algorithm_type (str): Either 'FFT', 'DFT 2D', 'DFT 3D' or
                'W-projection'.
        """
        self.capsule_ensure()
        _imager_lib.set_algorithm(self._capsule, algorithm_type)

    def set_cellsize(self, cellsize_arcsec):
        """Sets the cell (pixel) size.

        This can be used instead of set_fov() if required.
        After calling this method, changing the image size
        will change the field of view.

        Args:
            cellsize_arcsec (float): Image cell size, in arcsec.
        """
        self.capsule_ensure()
        _imager_lib.set_cellsize(self._capsule, cellsize_arcsec)

    def set_channel_snapshots(self, value):
        """Sets the flag specifying whether to image each channel separately.

        Args:
            value (boolean): If true, image each channel separately;
                if false, use frequency synthesis.
        """
        self.capsule_ensure()
        _imager_lib.set_channel_snapshots(self._capsule, value)

    def set_coords_only(self, flag):
        """Sets the imager to ignore visibility data and use coordinates only.

        Use this method with uniform weighting or W-projection.
        The grids of weights can only be used once they are fully populated,
        so this method puts the imager into a mode where it only updates its
        internal weights grids when calling update().

        This should only be called after setting all imager options.

        Turn this mode off when processing visibilities.

        Args:
            flag (boolean):
                If true, ignore visibility data and use coordinates only.
        """
        self.capsule_ensure()
        _imager_lib.set_coords_only(self._capsule, flag)

    def set_default_direction(self):
        """Clears any direction override."""
        self.capsule_ensure()
        _imager_lib.set_default_direction(self._capsule)

    def set_direction(self, ra_deg, dec_deg):
        """Sets the image centre different to the observation phase centre.

        Args:
            ra_deg (float): The new image Right Ascension, in degrees.
            dec_deg (float): The new image Declination, in degrees.
        """
        self.capsule_ensure()
        _imager_lib.set_direction(self._capsule, ra_deg, dec_deg)

    def set_fft_on_gpu(self, value):
        """Sets whether to use the GPU for FFTs.

        Args:
            value (boolean): If true, use the GPU for FFTs.
        """
        self.capsule_ensure()
        _imager_lib.set_fft_on_gpu(self._capsule, value)

    def set_fov(self, fov_deg):
        """Sets the field of view to image.

        This can be used instead of set_cellsize() if required.
        After calling this method, changing the image size
        will change the image resolution.

        Args:
            fov_deg (float): Field of view, in degrees.
        """
        self.capsule_ensure()
        _imager_lib.set_fov(self._capsule, fov_deg)

    def set_freq_max_hz(self, value):
        """Sets the maximum frequency of visibility data to image.

        A value less than or equal to zero means no maximum.

        Args:
            value (float):
                Maximum frequency of visibility data to image, in Hz.
        """
        self.capsule_ensure()
        _imager_lib.set_freq_max_hz(self._capsule, value)

    def set_freq_min_hz(self, value):
        """Sets the minimum frequency of visibility data to image.

        Args:
            value (float):
                Minimum frequency of visibility data to image, in Hz.
        """
        self.capsule_ensure()
        _imager_lib.set_freq_min_hz(self._capsule, value)

    def set_generate_w_kernels_on_gpu(self, value):
        """Sets whether to use the GPU to generate W-kernels.

        Args:
            value (boolean): If true, use the GPU to generate W-kernels.
        """
        self.capsule_ensure()
        _imager_lib.set_generate_w_kernels_on_gpu(self._capsule, value)

    def set_grid_kernel(self, kernel_type, support, oversample):
        """Sets the convolution kernel used for gridding visibilities.

        Args:
            kernel_type (str): Type of convolution kernel;
                either 'Spheroidal' or 'Pillbox'.
            support (int): Support size of kernel.
                The kernel width is 2 * support + 1.
            oversample (int): Oversample factor used for look-up table.
        """
        self.capsule_ensure()
        _imager_lib.set_grid_kernel(self._capsule, kernel_type,
                                    support, oversample)

    def set_image_size(self, size):
        """Sets image side length.

        Args:
            size (int): Image side length in pixels.
        """
        self.capsule_ensure()
        self.set_size(size)

    def set_image_type(self, image_type):
        """Sets the image (polarisation) type.

        Args:
            image_type (str): Either 'STOKES', 'I', 'Q', 'U', 'V',
                'LINEAR', 'XX', 'XY', 'YX', 'YY' or 'PSF'.
        """
        self.capsule_ensure()
        _imager_lib.set_image_type(self._capsule, image_type)

    def set_input_file(self, filename):
        """Sets the input visibility file or Measurement Set.

        Args:
            filename (str):
                Path to input Measurement Set or OSKAR visibility file.
        """
        self.capsule_ensure()
        _imager_lib.set_input_file(self._capsule, filename)

    def set_ms_column(self, column):
        """Sets the data column to use from a Measurement Set.

        Args:
            column (str): Name of the column to use.
        """
        self.capsule_ensure()
        _imager_lib.set_ms_column(self._capsule, column)

    def set_num_w_planes(self, num_planes):
        """Sets the number of W-planes to use, if using W-projection.

        A number less than or equal to zero means 'automatic'.

        Args:
            num_planes (int): Number of W-planes to use.
        """
        self.capsule_ensure()
        _imager_lib.set_num_w_planes(self._capsule, num_planes)

    def set_output_root(self, filename):
        """Sets the root path of output images.

        Args:
            filename (str): Root path.
        """
        self.capsule_ensure()
        _imager_lib.set_output_root(self._capsule, filename)

    def set_scale_norm_with_num_input_files(self, value):
        """Sets the option to scale image normalisation with number of files.

        Set this to true if the different files represent multiple
        sky model components observed with the same telescope configuration
        and observation parameters.
        Set this to false if the different files represent multiple
        observations of the same sky with different telescope configurations
        or observation parameters.

        Args:
            value (boolean): Option value.
        """
        self.capsule_ensure()
        _imager_lib.set_scale_norm_with_num_input_files(self._capsule, value)

    def set_size(self, size):
        """Sets image side length.

        Args:
            size (int): Image side length in pixels.
        """
        self.capsule_ensure()
        _imager_lib.set_size(self._capsule, size)

    def set_time_max_utc(self, value):
        """Sets the maximum time of visibility data to include in the image.

        A value less than or equal to zero means no maximum.

        Args:
            value (float): The maximum time of visibility data, as MJD(UTC).
        """
        self.capsule_ensure()
        _imager_lib.set_time_max_utc(self._capsule, value)

    def set_time_min_utc(self, value):
        """Sets the minimum time of visibility data to include in the image.

        Args:
            value (float): The minimum time of visibility data, as MJD(UTC).
        """
        self.capsule_ensure()
        _imager_lib.set_time_min_utc(self._capsule, value)

    def set_uv_filter_max(self, max_wavelength):
        """Sets the maximum UV baseline length to image, in wavelengths.

        A value less than zero means no maximum (i.e. all baseline
        lengths are allowed).

        Args:
            max_wavelength (float): Maximum UV distance, in wavelengths.
        """
        self.capsule_ensure()
        _imager_lib.set_uv_filter_max(self._capsule, max_wavelength)

    def set_uv_filter_min(self, min_wavelength):
        """Sets the minimum UV baseline length to image, in wavelengths.

        Args:
            min_wavelength (float): Minimum UV distance, in wavelengths.
        """
        self.capsule_ensure()
        _imager_lib.set_uv_filter_min(self._capsule, min_wavelength)

    def set_vis_frequency(self, ref_hz, inc_hz=0.0, num_channels=1):
        """Sets the visibility start frequency.

        Args:
            ref_hz (float):
                Frequency of channel index 0, in Hz.
            inc_hz (Optional[float]):
                Frequency increment, in Hz. Default 0.0.
            num_channels (Optional[int]):
                Number of channels in visibility data. Default 1.
        """
        self.capsule_ensure()
        _imager_lib.set_vis_frequency(self._capsule,
                                      ref_hz, inc_hz, num_channels)

    def set_vis_phase_centre(self, ra_deg, dec_deg):
        """Sets the coordinates of the visibility phase centre.

        Args:
            ra_deg (float): Right Ascension of phase centre, in degrees.
            dec_deg (float): Declination of phase centre, in degrees.
        """
        self.capsule_ensure()
        _imager_lib.set_vis_phase_centre(self._capsule, ra_deg, dec_deg)

    def set_weighting(self, weighting):
        """Sets the type of visibility weighting to use.

        Args:
            weighting (str): Either 'Natural', 'Radial' or 'Uniform'.
        """
        self.capsule_ensure()
        _imager_lib.set_weighting(self._capsule, weighting)

    def update(self, uu, vv, ww, amps=None, weight=None, time_centroid=None,
               start_channel=0, end_channel=0, num_pols=1):
        """Runs imager for supplied visibilities, applying optional selection.

        The visibility meta-data must be set prior to calling this method
        using set_vis_* methods.

        The visibility amplitude data dimension order must be:
        (slowest) time/baseline, channel, polarisation (fastest).
        This order is the same as that stored in a Measurement Set.

        The visibility weight data dimension order must be:
        (slowest) time/baseline, polarisation (fastest).

        If not given, the weights will be treated as all 1.

        The time_centroid parameter may be None if time filtering is not
        required.

        Call finalise() to finalise the images after calling this function.

        Args:
            uu (float, array-like, shape (n,)):
                Visibility uu coordinates, in metres.
            vv (float, array-like, shape (n,)):
                Visibility vv coordinates, in metres.
            ww (float, array-like, shape (n,)):
                Visibility ww coordinates, in metres.
            amps (Optional[complex float, array-like, shape (m,)]):
                Visibility complex amplitudes. Shape as described above.
            weight (Optional[float, array-like, shape (p,)]):
                Visibility weights. Shape as described above.
            time_centroid (Optional[float, array-like, shape (n,)]):
                Visibility time centroid values, as MJD(UTC) seconds.
            start_channel (Optional[int]):
                Start channel index of the visibility block. Default 0.
            end_channel (Optional[int]):
                End channel index of the visibility block. Default 0.
            num_pols (Optional[int]):
                Number of polarisations in the visibility block. Default 1.
        """
        self.capsule_ensure()
        _imager_lib.update(self._capsule, uu, vv, ww, amps, weight,
                           time_centroid, start_channel, end_channel, num_pols)

    def update_from_block(self, header, block):
        """Runs imager for visibility block, applying optional selection.

        Call finalise() to finalise the images after calling this function.

        Args:
            header (oskar.VisHeader): OSKAR visibility header.
            block (oskar.VisBlock):   OSKAR visibility block.
        """
        self.capsule_ensure()
        _imager_lib.update_from_block(self._capsule, header.capsule,
                                      block.capsule)

    def update_plane(self, uu, vv, ww, amps, weight, plane, plane_norm,
                     weights_grid=None):
        """Updates the supplied plane with the supplied visibilities.

        This is a low-level function that can be used to generate
        gridded visibilities if required.

        Visibility selection/filtering and phase rotation are
        not available at this level.

        When using a DFT, plane refers to the image plane;
        otherwise, plane refers to the visibility grid.

        The supplied baseline coordinates must be in wavelengths.

        If this is called in "coordinate only" mode, then the visibility
        amplitudes are ignored, the plane is untouched and the weights grid
        is updated instead.

        If the weight parameter is None, the weights will be treated as all 1.

        Call finalise_plane() to finalise the image after calling this
        function.

        Args:
            uu (float, array-like, shape (n,)):
                Visibility uu coordinates, in wavelengths.
            vv (float, array-like, shape (n,)):
                Visibility vv coordinates, in wavelengths.
            ww (float, array-like, shape (n,)):
                Visibility ww coordinates, in wavelengths.
            amps (complex float, array-like, shape (n,) or None):
                Visibility complex amplitudes.
            weight (float, array-like, shape (n,) or None):
                Visibility weights.
            plane (float, array-like or None):
                Plane to update.
            plane_norm (float):
                Current plane normalisation.
            weights_grid (Optional[float, array-like]):
                Gridded weights, size and shape of the grid plane.
                Used for uniform weighting.

        Returns:
            float: Updated plane normalisation.
        """
        self.capsule_ensure()
        return _imager_lib.update_plane(self._capsule, uu, vv, ww, amps,
                                        weight, plane, plane_norm,
                                        weights_grid)

    # Properties.
    algorithm = property(get_algorithm, set_algorithm)
    capsule = property(capsule_get, capsule_set)
    cell = property(get_cellsize, set_cellsize)
    cellsize = property(get_cellsize, set_cellsize)
    cellsize_arcsec = property(get_cellsize, set_cellsize)
    cell_size = property(get_cellsize, set_cellsize)
    cell_size_arcsec = property(get_cellsize, set_cellsize)
    channel_snapshots = property(get_channel_snapshots,
                                 set_channel_snapshots)
    coords_only = property(get_coords_only, set_coords_only)
    fft_on_gpu = property(get_fft_on_gpu, set_fft_on_gpu)
    fov = property(get_fov, set_fov)
    fov_deg = property(get_fov, set_fov)
    freq_max_hz = property(get_freq_max_hz, set_freq_max_hz)
    freq_min_hz = property(get_freq_min_hz, set_freq_min_hz)
    generate_w_kernels_on_gpu = property(get_generate_w_kernels_on_gpu,
                                         set_generate_w_kernels_on_gpu)
    image_size = property(get_image_size, set_image_size)
    image_type = property(get_image_type, set_image_type)
    input_file = property(get_input_file, set_input_file)
    input_files = property(get_input_file, set_input_file)
    input_vis_data = property(get_input_file, set_input_file)
    ms_column = property(get_ms_column, set_ms_column)
    num_w_planes = property(get_num_w_planes, set_num_w_planes)
    output_root = property(get_output_root, set_output_root)
    plane_size = property(get_plane_size)
    root_path = property(get_output_root, set_output_root)
    scale_norm_with_num_input_files = \
        property(get_scale_norm_with_num_input_files,
                 set_scale_norm_with_num_input_files)
    size = property(get_size, set_size)
    time_max_utc = property(get_time_max_utc, set_time_max_utc)
    time_min_utc = property(get_time_min_utc, set_time_min_utc)
    uv_filter_max = property(get_uv_filter_max, set_uv_filter_max)
    uv_filter_min = property(get_uv_filter_min, set_uv_filter_min)
    weighting = property(get_weighting, set_weighting)
    wprojplanes = property(get_num_w_planes, set_num_w_planes)

    @staticmethod
    def cellsize_to_fov(cellsize_rad, size):
        """Convert image cellsize and size along one dimension in pixels to FoV.

        Args:
            cellsize_rad (float): Image cell size, in radians.
            size (int):           Image size in one dimension in pixels.

        Returns:
            float: Image field-of-view, in radians.
        """
        return 2.0 * math.asin(0.5 * size * math.sin(cellsize_rad))

    @staticmethod
    def fov_to_cellsize(fov_rad, size):
        """Convert image FoV and size along one dimension in pixels to cellsize.

        Args:
            fov_rad (float):      Image field-of-view, in radians.
            size (int):           Image size in one dimension in pixels.

        Returns:
            float: Image cellsize, in radians.
        """
        return math.asin(2.0 * math.sin(0.5 * fov_rad) / size)

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

    @staticmethod
    def uv_cellsize_to_fov(uv_cellsize, size):
        """Convert a uv cellsize in wavelengths to an image FoV, in radians.

        uv cellsize is the size of a uv grid pixel in the grid used when
        imaging using an FFT.

        Args:
            uv_cellsize (float): Grid pixel size in wavelengths
            size (int): Size of the uv grid in pixels.

        Returns:
            float: Image field-of-view, in radians
        """
        return Imager.cellsize_to_fov(1 / (size * uv_cellsize), size)

    @staticmethod
    def extent_pixels(size):
        """Obtain the image or grid extent in pixel space

        Args:
            size (int): Image or grid size (number of pixels)

        Returns:
            numpy.ndarray: Image or grid extent in pixels ordered as follows
            [x_min, x_max, y_min, y_max]
        """
        c = size // 2
        return numpy.array([c + 0.5, -c + 0.5, -c - 0.5, c - 0.5])

    @staticmethod
    def image_extent_lm(fov_deg, size):
        """Return the image extent in direction cosines.

        The image extent is a list of 4 elements describing the
        dimensions of the image in the x (l) and y (m) axes.
        This can be used for example, to label images produced using the OSKAR
        imager when plotted with matplotlib's imshow() method.

        Args:
            fov_deg (float): Image field-of-view, in degrees
            size (int): Image size (number of pixels)

        Returns:
            numpy.ndarray: Image extent in direction cosines ordered as follows
            [l_min, l_max, m_min, m_max]
        """
        extent = numpy.array(Imager.extent_pixels(size))
        cellsize_rad = Imager.fov_to_cellsize(math.radians(fov_deg), size)
        cellsize_lm = math.sin(cellsize_rad)
        extent *= cellsize_lm
        return extent

    @staticmethod
    def grid_extent_wavelengths(fov_deg, size):
        """Return the the uv grid extent in wavelengths.

        The grid extent is a list of 4 elements describing the
        dimensions of the uv grid in the x (uu) and y (vv) axes.
        This can be used for example, to label uv grid images produced
        using the OSKAR imager when plotted with matplotlib's imshow() method.

        Args:
            fov_deg (float): Image field-of-view, in degrees
            size (int): Grid / image size (number of pixels)

        Returns:
            numpy.ndarray: Grid extent in wavelengths ordered as follows
            [uu_min, uu_max, vv_min, vv_max]
        """
        cellsize = Imager.fov_to_uv_cellsize(math.radians(fov_deg), size)
        extent = Imager.extent_pixels(size) * cellsize
        return extent

    @staticmethod
    def grid_pixels(grid_cellsize, size):
        """Return grid pixel coordinates the same units as grid_cellsize.

        Returns the x and y coordinates of the uv grid pixels for a
        grid / image of size x size pixels where the grid pixel separation is
        given by grid_cellsize. The output pixel coordinates will be in the
        same units as that of the supplied grid_cellsize.

        Args:
            grid_cellsize (float): Pixel separation in the grid
            size (int): Size of the grid / image

        Returns:
            tupple(gx, gy): where gx and gy are the pixel coordinates of each
            grid cell. gx and gy are 2d arrays of dimensions size x size.
        """
        x = numpy.arange(-size // 2, size // 2) * grid_cellsize
        gx, gy = numpy.meshgrid(-x, x)
        return gx, gy

    @staticmethod
    def image_pixels(fov_deg, im_size):
        """Return image pixel coordinates in lm (direction cosine) space.

        Args:
            fov_deg: Image field-of-view in degrees
            im_size: Image size in pixels.

        Returns:
            tupple (l, m): where l and m are the coordinates of each
            image pixel in the l (x) and m (y) directions. l and m are
            2d arrays of dimensions im_size by im_size
        """
        cell_size_rad = Imager.fov_to_cellsize(math.radians(fov_deg), im_size)
        cell_size_lm = math.sin(cell_size_rad)
        x = numpy.arange(-im_size // 2, im_size // 2) * cell_size_lm
        l, m = numpy.meshgrid(-x, x)
        return l, m

    @staticmethod
    def make_image(uu, vv, ww, amps, fov_deg, size, weighting='Natural',
                   algorithm='FFT', weight=None, wprojplanes=0):
        """Makes an image from visibility data.

        Args:
            uu (float, array-like, shape (n,)):
                Baseline uu coordinates, in wavelengths.
            vv (float, array-like, shape (n,)):
                Baseline vv coordinates, in wavelengths.
            ww (float, array-like, shape (n,)):
                Baseline ww coordinates, in wavelengths.
            amps (complex float, array-like, shape (n,)):
                Baseline visibility amplitudes.
            fov_deg (float): Image field of view, in degrees.
            size (int):      Image size along one dimension, in pixels.
            weighting (Optional[str]):
                Either 'Natural', 'Radial' or 'Uniform'.
            algorithm (Optional[str]):
                Algorithm type: 'FFT', 'DFT 2D', 'DFT 3D' or 'W-projection'.
            weight (Optional[float, array-like, shape (n,)]):
                Visibility weights.
            wprojplanes (Optional[int]):
                Number of W-projection planes to use, if using W-projection.
                If <= 0, this will be determined automatically.
                It will not be less than 16.

        Returns:
            array: Image as a 2D numpy array.
                Data are ordered as in FITS image.
        """
        if _imager_lib is None:
            raise RuntimeError("OSKAR library not found.")
        return _imager_lib.make_image(uu, vv, ww, amps, fov_deg, size,
                                      weighting, algorithm, weight,
                                      wprojplanes)
