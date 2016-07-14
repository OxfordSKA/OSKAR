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
        self._capsule = _imager_lib.create(precision)


    def check_init(self):
        """Initialises the imager algorithm if it has not already been done.

        All imager options and data must have been set appropriately
        before calling this function.
        """
        _imager_lib.check_init(self._capsule)


    def finalise(self, image=None):
        """Finalises the image or images and writes them to file.

        Args:
            image (Optional[float, array-like]):
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
            plane (complex float, array-like):
                On input, the plane to finalise; on output, the image plane.
            plane_norm (float): Plane normalisation to apply.
        """
        _imager_lib.finalise_plane(self._capsule, plane, plane_norm)


    def get_algorithm(self):
        """Returns a string describing the imager algorithm.

        Returns:
            str: The imager algorithm.
        """
        return _imager_lib.algorithm(self._capsule)


    def get_coords_only(self):
        """Returns flag specifying whether imager is in coordinate-only mode.

        Returns:
            bool: If true, imager is in coordinate-only mode.
        """
        return _imager_lib.coords_only(self._capsule)


    def get_fov(self):
        """Returns the image field-of-view, in degrees.

        Returns:
            double: The image field-of-view, in degrees.
        """
        return _imager_lib.fov(self._capsule)


    def get_image_size(self):
        """Returns the image side length, in pixels.

        Returns:
            int: The image side length, in pixels.
        """
        return _imager_lib.image_size(self._capsule)


    def get_image_type(self):
        """Returns a string describing the image (polarisation) type.

        Returns:
            str: The image (polarisation) type.
        """
        return _imager_lib.image_type(self._capsule)


    def get_input_file(self):
        """Returns a string containing the input file name.

        Returns:
            str: The input file name.
        """
        return _imager_lib.input_file(self._capsule)


    def get_ms_column(self):
        """Returns a string containing the Measurement Set column to use.

        Returns:
            str: The column name.
        """
        return _imager_lib.ms_column(self._capsule)


    def get_num_w_planes(self):
        """Returns the number of W-planes used.

        Returns:
            int: The number of W-planes used.
        """
        return _imager_lib.num_w_planes(self._capsule)


    def get_output_root(self):
        """Returns a string containing the output root file name.

        Returns:
            str: The output root file name.
        """
        return _imager_lib.output_root(self._capsule)


    def get_plane_size(self):
        """Returns the required plane size.

        This may be different to the image size, for example if using 
        W-projection. It will only be valid after a call to check_init().

        Returns:
            int: Plane side length.
        """
        return _imager_lib.plane_size(self._capsule)


    def get_size(self):
        """Returns the image side length, in pixels.

        Returns:
            int: The image side length, in pixels.
        """
        return _imager_lib.size(self._capsule)


    def get_weighting(self):
        """Returns a string describing the weighting scheme.

        Returns:
            str: The weighting scheme.
        """
        return _imager_lib.weighting(self._capsule)


    def reset_cache(self):
        """Low-level function to reset the imager's internal memory.

        This is used to clear any data added using update().
        """
        _imager_lib.reset_cache(self._capsule)


    def run(self):
        """Runs the imager on a visibility file.

        The input filename must be set using set_input_file().
        """
        _imager_lib.run(self._capsule)


    def set(self, **kwargs):
        """Sets multiple properties at once.

        For example: set(fov=2.0, image_size=2048, algorithm='W-projection')
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


    def set_algorithm(self, algorithm_type):
        """Sets the algorithm used by the imager.

        Args:
            type (str): Either 'FFT', 'DFT 2D', 'DFT 3D' or 'W-projection'.
        """
        _imager_lib.set_algorithm(self._capsule, algorithm_type)


    def set_channel_range(self, start, end, snapshots):
        """Sets the visibility channel range used by the imager.

        Args:
            start (int): Start channel index.
            end (int):   End channel index (-1 for all channels).
            snapshots (boolean): If true, image each channel separately; 
                if false, use frequency synthesis.
        """
        _imager_lib.set_channel_range(self._capsule, start, end, snapshots)


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
        _imager_lib.set_coords_only(self._capsule, flag)


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


    def set_grid_kernel(self, kernel_type, support, oversample):
        """Sets the convolution kernel used for gridding visibilities.

        Args:
            type (str): Type of convolution kernel; 
                either 'Spheroidal' or 'Pillbox'.
            support (int): Support size of kernel. 
                The kernel width is 2 * support + 1.
            oversample (int): Oversample factor used for look-up table.
        """
        _imager_lib.set_grid_kernel(self._capsule, kernel_type,
            support, oversample)


    def set_image_size(self, size):
        """Sets image side length.

        Args:
            size (int): Image side length in pixels.
        """
        self.set_size(size)


    def set_image_type(self, image_type):
        """Sets the image (polarisation) type.

        Args:
            type (str): Either 'STOKES', 'I', 'Q', 'U', 'V', 
                'LINEAR', 'XX', 'XY', 'YX', 'YY' or 'PSF'.
        """
        _imager_lib.set_image_type(self._capsule, image_type)


    def set_input_file(self, filename):
        """Sets the input visibility file or Measurement Set.

        Args:
            filename (str): 
                Path to input Measurement Set or OSKAR visibility file.
        """
        _imager_lib.set_input_file(self._capsule, filename)


    def set_ms_column(self, column):
        """Sets the data column to use from a Measurement Set.

        Args:
            column (str): Name of the column to use.
        """
        _imager_lib.set_ms_column(self._capsule, column)


    def set_num_w_planes(self, num_planes):
        """Sets the number of W-planes to use, if using W-projection.

        A number less than or equal to zero means 'automatic'.

        Args:
            num_planes (int): Number of W-planes to use.
        """
        _imager_lib.set_num_w_planes(self._capsule, num_planes)


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
        if size % 2 != 0:
            raise RuntimeError("Image size must be even.")
            return
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


    def set_vis_frequency(self, ref_hz, inc_hz=0.0, num_channels=1):
        """Sets the visibility start frequency.

        Args:
            ref_hz (float):
                Frequency of index 0, in Hz.
            inc_hz (Optional[float]):
                Frequency increment, in Hz. Default 0.0.
            num_channels (Optional[int]):
                Number of channels in visibility data. Default 1.
        """
        _imager_lib.set_vis_frequency(self._capsule,
            ref_hz, inc_hz, num_channels)


    def set_vis_phase_centre(self, ra_deg, dec_deg):
        """Sets the coordinates of the visibility phase centre.

        Args:
            ra_deg (float): Right Ascension of phase centre, in degrees.
            dec_deg (float): Declination of phase centre, in degrees.
        """
        _imager_lib.set_vis_phase_centre(self._capsule, ra_deg, dec_deg)


    def set_vis_time(self, ref_mjd_utc, inc_sec=0.0, num_times=1):
        """Sets the visibility start time.

        Args:
            ref_mjd_utc (float):
                Time of index 0, as MJD(UTC).
            inc_sec (Optional[float]):
                Time increment, in seconds. Default 0.0.
            num_times (Optional[int]):
                Number of time steps in visibility data.
        """
        _imager_lib.set_vis_time(self._capsule, ref_mjd_utc, inc_sec, num_times)


    def set_weighting(self, weighting):
        """Sets the type of visibility weighting to use.

        Args:
            weighting (str): Either 'Natural', 'Radial' or 'Uniform'.
        """
        _imager_lib.set_weighting(self._capsule, weighting)


    def update(self, num_baselines, uu, vv, ww, amps, weight, num_pols=1,
            start_time=0, end_time=0, start_channel=0, end_channel=0):
        """Runs imager for supplied visibilities, applying optional selection.

        The visibility amplitude data dimension order must be:
        (slowest) time, channel, baseline, polarisation (fastest).

        The visibility weight data dimension order must be:
        (slowest) time, baseline, polarisation (fastest).

        Call finalise() to finalise the images after calling this function.

        Args:
            num_baselines (int):
                Number of baselines in the visibility block.
            uu (float, array-like, shape (n,)):
                Time-baseline ordered uu coordinates, in metres.
            vv (float, array-like, shape (n,)):
                Time-baseline ordered vv coordinates, in metres.
            ww (float, array-like, shape (n,)):
                Time-baseline ordered ww coordinates, in metres.
            amp (complex float, array-like, shape (n,)):
                Baseline visibility amplitudes. Length as described above.
            weight (float, array-like, shape (n,)):
                Visibility weights. Length as described above.
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


    def update_from_block(self, header, block):
        """Runs imager for visibility block, applying optional selection.

        Call finalise() to finalise the images after calling this function.

        Args:
            header (oskar.VisHeader):
                Handle to an OSKAR visibility header.
            block (oskar.VisBlock):
                Handle to an OSKAR visibility block.
        """
        _imager_lib.update_from_block(self._capsule, 
            header._capsule, block._capsule)


    def update_plane(self, uu, vv, ww, amps, weight, plane, plane_norm,
            weights_grid=None):
        """Updates the supplied plane with the supplied visibilities.

        This is a low-level function that can be used to generate 
        gridded visibilities if required.

        Call finalise_plane() to finalise the image after calling this function.

        Args:
            uu (float, array-like, shape (n,)):
                Baseline uu coordinates, in wavelengths.
            vv (float, array-like, shape (n,)):
                Baseline vv coordinates, in wavelengths.
            ww (float, array-like, shape (n,)):
                Baseline ww coordinates, in wavelengths.
            amps (complex float, array-like, shape (n,)):
                Baseline visibility amplitudes.
            weight (float, array-like, shape (n,)):
                Visibility weights.
            plane (float, array-like):
                Plane to update.
            plane_norm (float):
                Current plane normalisation.
            weights_grid (Optional[float, array-like]):
                Gridded weights, size and shape of the image plane.
                Used for uniform weighting.

        Returns:
            float: Updated plane normalisation.
        """
        return _imager_lib.update_plane(self._capsule, uu, vv, ww, 
            amps, weight, plane, plane_norm, weights_grid)


    # Properties.
    algorithm      = property(get_algorithm, set_algorithm)
    coords_only    = property(get_coords_only, set_coords_only)
    fov            = property(get_fov, set_fov)
    fov_deg        = property(get_fov, set_fov)
    image_size     = property(get_image_size, set_image_size)
    image_type     = property(get_image_type, set_image_type)
    input_file     = property(get_input_file, set_input_file)
    ms_column      = property(get_ms_column, set_ms_column)
    num_w_planes   = property(get_num_w_planes, set_num_w_planes)
    output_root    = property(get_output_root, set_output_root)
    plane_size     = property(get_plane_size)
    size           = property(get_size, set_size)
    weighting      = property(get_weighting, set_weighting)
    wprojplanes    = property(get_num_w_planes, set_num_w_planes)


    @staticmethod
    def make_image(uu, vv, ww, amps, fov_deg, size,
            weighting='Natural', algorithm='FFT', weight=None, wprojplanes=0):
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
            array: Image as a 2D numpy array. Data are ordered as in FITS image.
        """
        if size % 2 != 0:
            raise RuntimeError("Image size must be even.")
            return
        return _imager_lib.make_image(uu, vv, ww, amps, fov_deg, size,
            weighting, algorithm, weight, wprojplanes)


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

