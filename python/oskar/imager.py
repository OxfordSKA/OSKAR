# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2020, The University of Oxford
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

"""Interfaces to the OSKAR imager."""

from __future__ import absolute_import, division, print_function
import math
import numpy
try:
    from . import _imager_lib
except ImportError:
    _imager_lib = None

# pylint: disable=useless-object-inheritance,too-many-public-methods
# pylint: disable=invalid-name
class Imager(object):
    """This class provides a Python interface to the OSKAR imager.

    The :class:`oskar.Imager` class allows basic (dirty) images to be made
    from simulated visibility data, either from data files on disk, from
    numpy arrays in memory, or from :class:`oskar.VisBlock` objects.

    There are three stages required to make an image:

    1. Create and set-up an imager.
    2. Update the imager with visibility data, possibly multiple times.
    3. Finalise the imager to generate the image.

    The last two stages can be combined for ease of use if required, simply by
    calling the :meth:`run() <oskar.Imager.run()>` method with no arguments.

    To set up the imager from Python, create an instance of the class and set
    the imaging options using a :class:`oskar.SettingsTree` created for the
    ``oskar_imager`` application, and/or set properties on the class itself.
    These include:

    - :meth:`algorithm <oskar.Imager.algorithm>` and
      :meth:`weighting <oskar.Imager.weighting>`
    - :meth:`cellsize_arcsec <oskar.Imager.cellsize_arcsec>` or
      :meth:`fov_deg <oskar.Imager.fov_deg>`
    - :meth:`channel_snapshots <oskar.Imager.channel_snapshots>`
    - :meth:`fft_on_gpu <oskar.Imager.fft_on_gpu>` and
      :meth:`grid_on_gpu <oskar.Imager.grid_on_gpu>`
    - :meth:`image_size <oskar.Imager.image_size>`
    - :meth:`image_type <oskar.Imager.image_type>`

    To optionally filter the input visibility data, use:

    - :meth:`freq_max_hz <oskar.Imager.freq_max_hz>` and
      :meth:`freq_min_hz <oskar.Imager.freq_min_hz>` to exclude visibilities
      based on frequency.
    - :meth:`time_max_utc <oskar.Imager.time_max_utc>` and
      :meth:`time_min_utc <oskar.Imager.time_min_utc>` to exclude visibilities
      based on time.
    - :meth:`uv_filter_max <oskar.Imager.uv_filter_max>` and
      :meth:`uv_filter_min <oskar.Imager.uv_filter_min>` to exclude
      visibilities based on their (u,v)-baseline length in wavelengths.

    To specify input and/or output files, use
    :meth:`input_file <oskar.Imager.input_file>` and
    :meth:`output_root <oskar.Imager.output_root>`.

    For convenience, the :meth:`set() <oskar.Imager.set>` method can be used
    to set multiple properties at once using ``kwargs``.

    Off-phase-centre imaging is supported. Use the
    :meth:`set_direction() <oskar.Imager.set_direction>` method to centre the
    image around different coordinates if required.

    When imaging visibility data from a file, it is sufficient simply to call
    the :meth:`run() <oskar.Imager.run>` method with no arguments.
    However, it is often necessary to process visibility data prior to imaging
    it (perhaps by subtracting model visibilities), and for this reason it may
    be more useful to pass the visibility data to the imager explicitly either
    via parameters to :meth:`run() <oskar.Imager.run>` (which will also
    finalise the image) or using the :meth:`update() <oskar.Imager.update>`
    method (which may be called multiple times if necessary).
    The convenience method
    :meth:`update_from_block() <oskar.Imager.update_from_block>` can be used
    instead if visibility data are contained within a :class:`oskar.VisBlock`.

    If passing numpy arrays to :meth:`run() <oskar.Imager.run>` or
    :meth:`update() <oskar.Imager.update>`, be sure to set the frequency and
    phase centre first using
    :meth:`set_vis_frequency() <oskar.Imager.set_vis_frequency>` and
    :meth:`set_vis_phase_centre() <oskar.Imager.set_vis_phase_centre>`.

    After all visibilities have been processed using
    :meth:`update() <oskar.Imager.update>` or
    :meth:`update_from_block() <oskar.Imager.update_from_block>`, call
    :meth:`finalise() <oskar.Imager.finalise>` to generate the image.
    The images and/or gridded visibilities can be returned directly to
    Python as numpy arrays if required.

    Note that uniform weighting requires all visibility coordinates to be
    known in advance. To allow for this, set the
    :meth:`coords_only <oskar.Imager.coords_only>` property to ``True`` to
    switch the imager into a "coordinates-only" mode before calling
    :meth:`update() <oskar.Imager.update>`. Once all the coordinates have been
    read, set :meth:`coords_only <oskar.Imager.coords_only>` to ``False``,
    initialise the imager by calling
    :meth:`check_init() <oskar.Imager.check_init>` and then call
    :meth:`update() <oskar.Imager.update>` again.
    (This is done automatically if using :meth:`run() <oskar.Imager.run>`
    instead.)

    Examples:

        >>> # Generate some data to process.
        >>> import numpy
        >>> n = 100000  # Number of visibility points.
        >>> # This will generate a filled circular aperture.
        >>> t = 2 * numpy.pi * numpy.random.random(n)
        >>> r = 50e3 * numpy.sqrt(numpy.random.random(n))
        >>> uu = r * numpy.cos(t)
        >>> vv = r * numpy.sin(t)
        >>> ww = numpy.zeros_like(uu)
        >>> vis = numpy.ones(n, dtype='c16')  # Point source at phase centre.

        To make an image using supplied (u,v,w) coordinates and visibilities,
        and return the image to Python:

        >>> # (continued from previous section)
        >>> imager = oskar.Imager()
        >>> imager.fov_deg = 0.1             # 0.1 degrees across.
        >>> imager.image_size = 256          # 256 pixels across.
        >>> imager.set_vis_frequency(100e6)  # 100 MHz, single channel data.
        >>> imager.update(uu, vv, ww, vis)
        >>> data = imager.finalise(return_images=1)
        >>> image = data['images'][0]

        To plot the image using matplotlib:

        >>> # (continued from previous section)
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(image)
        >>> plt.show()

        .. figure:: example_image1.png
           :width: 640px
           :align: center
           :height: 480px
           :alt: An example image of a point source,
                 generated using a filled aperture and plotted using matplotlib

           An example image of a point source,
           generated using a filled aperture and plotted using matplotlib.

    """

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

    @property
    def algorithm(self):
        """Returns or sets the algorithm used by the imager.
        Currently one of 'FFT', 'DFT 2D', 'DFT 3D' or 'W-projection'.

        The default is 'FFT', which corresponds to basic but quick 2D gridding,
        ignoring baseline w-components.
        DFT-based methods are (**very, very!**) slow but more accurate.
        The 'DFT 3D' option can be used to make a "perfect" undistorted
        dirty image, but should only be considered for use with small data sets.

        Use 'W-projection' (or your favourite other imaging tool -
        there are many!) to avoid distortions caused by imaging wide fields
        of view using non-coplanar baselines.
        The implementation of W-projection in OSKAR produces results consistent
        with dirty images from CASA, although the GPU version in OSKAR is
        considerably faster for large visibility data sets: use
        :meth:`grid_on_gpu <oskar.Imager.grid_on_gpu>` (and optionally
        :meth:`fft_on_gpu <oskar.Imager.fft_on_gpu>`) to enable it,
        but ensure you have enough GPU memory available for the size
        of image you are making, as an extra copy of the grid
        will be made by the FFT library.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.algorithm(self._capsule)

    @algorithm.setter
    def algorithm(self, value):
        self.set_algorithm(value)

    @property
    def cellsize_arcsec(self):
        """Returns or sets the cell (pixel) size, in arcsec.

        After setting this property, changing the
        :meth:`image size <oskar.Imager.image_size>`
        will change the field of view. Can be used instead of
        :meth:`fov_deg <oskar.Imager.fov_deg>` if required.

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.cellsize(self._capsule)

    @cellsize_arcsec.setter
    def cellsize_arcsec(self, value):
        self.set_cellsize(value)

    @property
    def channel_snapshots(self):
        """Returns or sets the flag to image channels separately.

        By default, this is false.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.channel_snapshots(self._capsule)

    @channel_snapshots.setter
    def channel_snapshots(self, value):
        self.set_channel_snapshots(value)

    @property
    def coords_only(self):
        """Returns or sets the flag to use coordinates only.

        Set this property when using uniform weighting or W-projection.
        The grids of weights can only be used once they are fully populated,
        so this method puts the imager into a mode where it only updates its
        internal weights grids when calling
        :meth:`update() <oskar.Imager.update>`.

        This should only be used after setting all imager options.

        Turn this mode off when processing visibilities,
        otherwise they will be ignored.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.coords_only(self._capsule)

    @coords_only.setter
    def coords_only(self, value):
        self.set_coords_only(value)

    @property
    def fft_on_gpu(self):
        """Returns or sets the flag to use the GPU for FFTs.

        By default, this is false.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.fft_on_gpu(self._capsule)

    @fft_on_gpu.setter
    def fft_on_gpu(self, value):
        self.set_fft_on_gpu(value)

    @property
    def fov_deg(self):
        """Returns or sets the image field-of-view, in degrees.

        After setting this property, changing the
        :meth:`image size <oskar.Imager.image_size>`
        will change the image resolution. Can be used instead of
        :meth:`cellsize_arcsec <oskar.Imager.cellsize_arcsec>` if required.

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.fov(self._capsule)

    @fov_deg.setter
    def fov_deg(self, value):
        self.set_fov(value)

    @property
    def freq_max_hz(self):
        """Returns or sets the maximum frequency of visibility data to image.

        A value less than or equal to zero means no maximum.

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.freq_max_hz(self._capsule)

    @freq_max_hz.setter
    def freq_max_hz(self, value):
        self.set_freq_max_hz(value)

    @property
    def freq_min_hz(self):
        """Returns or sets the minimum frequency of visibility data to image.

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.freq_min_hz(self._capsule)

    @freq_min_hz.setter
    def freq_min_hz(self, value):
        self.set_freq_min_hz(value)

    @property
    def generate_w_kernels_on_gpu(self):
        """Returns or sets the flag to use the GPU to generate kernels
        for W-projection.

        By default, this is true.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.generate_w_kernels_on_gpu(self._capsule)

    @generate_w_kernels_on_gpu.setter
    def generate_w_kernels_on_gpu(self, value):
        self.set_generate_w_kernels_on_gpu(value)

    @property
    def grid_on_gpu(self):
        """Returns or sets the flag to use the GPU for gridding.

        By default, this is false.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.grid_on_gpu(self._capsule)

    @grid_on_gpu.setter
    def grid_on_gpu(self, value):
        self.set_grid_on_gpu(value)

    @property
    def image_size(self):
        """Returns or sets the image side length in pixels.

        Type
            int
        """
        self.capsule_ensure()
        return _imager_lib.image_size(self._capsule)

    @image_size.setter
    def image_size(self, value):
        self.set_image_size(value)

    @property
    def image_type(self):
        """Returns or sets the image (polarisation) type.

        Either 'STOKES', 'I', 'Q', 'U', 'V',
        'LINEAR', 'XX', 'XY', 'YX', 'YY' or 'PSF'.

        By default, this is 'I' (for Stokes I only).

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.image_type(self._capsule)

    @image_type.setter
    def image_type(self, value):
        self.set_image_type(value)

    @property
    def input_file(self):
        """Returns or sets the input visibility file or Measurement Set.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.input_file(self._capsule)

    @input_file.setter
    def input_file(self, value):
        self.set_input_file(value)

    @property
    def ms_column(self):
        """Returns or sets the data column to use from a Measurement Set.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.ms_column(self._capsule)

    @ms_column.setter
    def ms_column(self, value):
        self.set_ms_column(value)

    @property
    def num_w_planes(self):
        """Returns or sets the number of W-projection planes to use,
        if using W-projection.

        A number less than or equal to zero means 'automatic'.

        Type
            int
        """
        self.capsule_ensure()
        return _imager_lib.num_w_planes(self._capsule)

    @num_w_planes.setter
    def num_w_planes(self, value):
        self.set_num_w_planes(value)

    @property
    def output_root(self):
        """Returns or sets the root path of output images.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.output_root(self._capsule)

    @output_root.setter
    def output_root(self, value):
        self.set_output_root(value)

    @property
    def plane_size(self):
        """Returns the required plane side length.

        This may be different to the image size, for example
        if using W-projection.
        It will only be valid after a call to
        :meth:`check_init() <oskar.Imager.check_init>`.

        Type
            int
        """
        self.capsule_ensure()
        return _imager_lib.plane_size(self._capsule)

    @property
    def root_path(self):
        """Returns or sets the root path of output images.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.output_root(self._capsule)

    @root_path.setter
    def root_path(self, value):
        self.set_output_root(value)

    @property
    def scale_norm_with_num_input_files(self):
        """Returns or sets the option to scale image normalisation with
        the number of files.

        Set this to true if the different files represent multiple
        sky model components observed with the same telescope configuration
        and observation parameters.
        Set this to false if the different files represent multiple
        observations of the same sky with different telescope configurations
        or observation parameters.

        By default, this is false.

        Type
            boolean
        """
        self.capsule_ensure()
        return _imager_lib.scale_norm_with_num_input_files(self._capsule)

    @scale_norm_with_num_input_files.setter
    def scale_norm_with_num_input_files(self, value):
        self.set_scale_norm_with_num_input_files(value)

    @property
    def size(self):
        """Returns or sets the image side length in pixels.

        Type
            int
        """
        self.capsule_ensure()
        return _imager_lib.size(self._capsule)

    @size.setter
    def size(self, value):
        self.set_size(value)

    @property
    def time_max_utc(self):
        """Returns or sets the maximum time of visibility data to image.

        A value less than or equal to zero means no maximum.
        The value is given as MJD(UTC).

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.time_max_utc(self._capsule)

    @time_max_utc.setter
    def time_max_utc(self, value):
        self.set_time_max_utc(value)

    @property
    def time_min_utc(self):
        """Returns or sets the minimum time of visibility data to image.

        The value is given as MJD(UTC).

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.time_min_utc(self._capsule)

    @time_min_utc.setter
    def time_min_utc(self, value):
        self.set_time_min_utc(value)

    @property
    def uv_filter_max(self):
        """Returns or sets the maximum UV baseline length to image,
        in wavelengths.

        A value less than zero means no maximum
        (i.e. all baseline lengths are allowed).

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.uv_filter_max(self._capsule)

    @uv_filter_max.setter
    def uv_filter_max(self, value):
        self.set_uv_filter_max(value)

    @property
    def uv_filter_min(self):
        """Returns or sets the minimum UV baseline length to image,
        in wavelengths.

        Type
            float
        """
        self.capsule_ensure()
        return _imager_lib.uv_filter_min(self._capsule)

    @uv_filter_min.setter
    def uv_filter_min(self, value):
        self.set_uv_filter_min(value)

    @property
    def weighting(self):
        """Returns or sets the type of visibility weighting to use.

        Either 'Natural', 'Radial' or 'Uniform'. The default is 'Natural'.

        Type
            str
        """
        self.capsule_ensure()
        return _imager_lib.weighting(self._capsule)

    @weighting.setter
    def weighting(self, value):
        self.set_weighting(value)

    @property
    def wprojplanes(self):
        """Returns or sets the number of W-projection planes to use,
        if using W-projection.

        A number less than or equal to zero means 'automatic'.

        Type
            int
        """
        self.capsule_ensure()
        return _imager_lib.num_w_planes(self._capsule)

    @wprojplanes.setter
    def wprojplanes(self, value):
        self.set_num_w_planes(value)

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
        Use e.g. ['images'][0] to access the first image.

        Args:
            return_images (Optional[int]): Number of image planes to return.
            return_grids (Optional[int]): Number of grid planes to return.

        Returns:
            dict: Python dictionary containing two keys, 'images' and 'grids',
            which are themselves arrays.
        """
        self.capsule_ensure()
        return _imager_lib.finalise(self._capsule, return_images, return_grids)

    def finalise_plane(self, plane, plane_norm):
        """Finalises an image plane.

        This is a low-level function that is used to finalise
        gridded visibilities when used in conjunction with
        :meth:`update_plane() <oskar.Imager.update_plane>`.

        The image can be obtained by taking the real part of the plane after
        this function returns.

        Args:
            plane (complex float, array-like):
                On input, the plane to finalise; on output, the image plane.
            plane_norm (float): Plane normalisation to apply.
        """
        self.capsule_ensure()
        _imager_lib.finalise_plane(self._capsule, plane, plane_norm)

    def reset_cache(self):
        """Low-level function to reset the imager's internal memory.

        This is used to clear any data added using
        :meth:`update() <oskar.Imager.update>`.
        """
        self.capsule_ensure()
        _imager_lib.reset_cache(self._capsule)

    def rotate_coords(self, uu_in, vv_in, ww_in):
        """Rotates baseline coordinates to the new phase centre (if set).

        Prior to calling this method, the new phase centre must be set first
        using :meth:`set_direction() <oskar.Imager.set_direction>`,
        and then the original phase centre must be set using
        :meth:`set_vis_phase_centre() <oskar.Imager.set_vis_phase_centre>`.
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
        using :meth:`set_direction() <oskar.Imager.set_direction>`,
        and then the original phase centre must be set using
        :meth:`set_vis_phase_centre() <oskar.Imager.set_vis_phase_centre>`.
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
        If using a file, the input filename must be set using
        :meth:`input_file <oskar.Imager.input_file>`.
        If using arrays, the visibility meta-data must be set prior to calling
        this method using
        :meth:`set_vis_frequency() <oskar.Imager.set_vis_frequency>` and
        :meth:`set_vis_phase_centre() <oskar.Imager.set_vis_phase_centre>`.

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
        Use e.g. ['images'][0] to access the first image.

        Args:
            uu (Optional[float, array-like, shape (n,)]):
                Time-baseline ordered uu coordinates, in metres.
            vv (Optional[float, array-like, shape (n,)]):
                Time-baseline ordered vv coordinates, in metres.
            ww (Optional[float, array-like, shape (n,)]):
                Time-baseline ordered ww coordinates, in metres.
            amps (Optional[complex float, array-like, shape (m,)]):
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

        Returns:
            dict: Python dictionary containing two keys, 'images' and 'grids',
            which are themselves arrays.
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

    def set_grid_on_gpu(self, value):
        """Sets whether to use the GPU for gridding.

        Args:
            value (boolean): If true, use the GPU for gridding.
        """
        self.capsule_ensure()
        _imager_lib.set_grid_on_gpu(self._capsule, value)

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

        The visibility meta-data must be set prior to calling
        this method using
        :meth:`set_vis_frequency() <oskar.Imager.set_vis_frequency>` and
        :meth:`set_vis_phase_centre() <oskar.Imager.set_vis_phase_centre>`.

        The visibility amplitude data dimension order must be:
        (slowest) time/baseline, channel, polarisation (fastest).
        This order is the same as that stored in a Measurement Set.

        The visibility weight data dimension order must be:
        (slowest) time/baseline, polarisation (fastest).

        If not given, the weights will be treated as all 1.

        The time_centroid parameter may be None if time filtering is not
        required.

        Call :meth:`finalise() <oskar.Imager.finalise>` to finalise the images
        after calling this function.

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

        Call :meth:`finalise() <oskar.Imager.finalise>` to finalise the images
        after calling this function.

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

        Call :meth:`finalise_plane() <oskar.Imager.finalise_plane>` to finalise
        the image after calling this function.

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
            tuple: The tuple elements gx and gy are the pixel coordinates of
            each grid cell. gx and gy are 2D arrays of dimensions size by size.
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
            tuple: The tuple elements l and m are the coordinates of each
            image pixel in the l (x) and m (y) directions. l and m are
            2D arrays of dimensions im_size by im_size
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

        Convenience static function.

        Args:
            uu (float, array-like, shape (n,)):
                Baseline uu coordinates, in wavelengths.
            vv (float, array-like, shape (n,)):
                Baseline vv coordinates, in wavelengths.
            ww (float, array-like, shape (n,)):
                Baseline ww coordinates, in wavelengths.
            amps (complex float, array-like, shape (n,)):
                Baseline visibility amplitudes.
            fov_deg (float):
                Image field of view, in degrees.
            size (int):
                Image size along one dimension, in pixels.
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
            numpy.ndarray: Image as a 2D numpy array.
            Data are ordered as in FITS image.
        """
        if _imager_lib is None:
            raise RuntimeError("OSKAR library not found.")
        return _imager_lib.make_image(uu, vv, ww, amps, fov_deg, size,
                                      weighting, algorithm, weight,
                                      wprojplanes)

    capsule = property(capsule_get, capsule_set)
    cell = property(cellsize_arcsec, set_cellsize)
    cellsize = property(cellsize_arcsec, set_cellsize)
    cell_size = property(cellsize_arcsec, set_cellsize)
    cell_size_arcsec = property(cellsize_arcsec, set_cellsize)
    fov = property(fov_deg, set_fov)
    input_files = property(input_file, set_input_file)
    input_vis_data = property(input_file, set_input_file)
