# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2020, The OSKAR Developers.
# See the LICENSE file at the top-level directory of this distribution.
#

"""Interfaces to the OSKAR sky model."""

from __future__ import absolute_import, division, print_function
import math
import numpy
try:
    from . import _sky_lib
except ImportError:
    _sky_lib = None


class Sky(object):
    """This class provides a Python interface to an OSKAR sky model.

    The :class:`oskar.Sky` class can be used to create an OSKAR sky model from
    Python. The sky model can be thought of as simply a table of source data,
    where each row of the table contains parameters for a single source.
    The sky model table format is described in the
    `sky model documentation <https://github.com/OxfordSKA/OSKAR/releases>`_
    and a summary of the columns is also included below for convenience.
    Only the first three are manadatory (others can be omitted, and will
    default to zero).

    +--------+---------------------+-----------+
    | Column | Parameter           | Unit      |
    +========+=====================+===========+
    | 1      | Right Ascension     | deg       |
    +--------+---------------------+-----------+
    | 2      | Declination         | deg       |
    +--------+---------------------+-----------+
    | 3      | Stokes I flux       | Jy        |
    +--------+---------------------+-----------+
    | 4      | Stokes Q flux       | Jy        |
    +--------+---------------------+-----------+
    | 5      | Stokes U flux       | Jy        |
    +--------+---------------------+-----------+
    | 6      | Stokes V flux       | Jy        |
    +--------+---------------------+-----------+
    | 7      | Reference frequency | Hz        |
    +--------+---------------------+-----------+
    | 8      | Spectral index      | N/A       |
    +--------+---------------------+-----------+
    | 9      | Rotation measure    | rad / m^2 |
    +--------+---------------------+-----------+
    | 10     | Major axis FWHM     | arcsec    |
    +--------+---------------------+-----------+
    | 11     | Minor axis FWHM     | arcsec    |
    +--------+---------------------+-----------+
    | 12     | Position angle      | deg       |
    +--------+---------------------+-----------+

    A sky model can be loaded from a text file using the
    :meth:`load() <oskar.Sky.load()>` method,
    or created directly from a numpy array using the
    :meth:`from_array() <oskar.Sky.from_array()>` method.
    Sky models can be concatenated using :meth:`append() <oskar.Sky.append()>`,
    which is useful if building up a sky model in multiple stages.

    Sky models can be filtered to exclude sources outside
    specified radii from a given point using the method
    :meth:`filter_by_radius() <oskar.Sky.filter_by_radius()>`,
    and filtered by Stokes-I flux using the method
    :meth:`filter_by_flux() <oskar.Sky.filter_by_flux()>`.

    The number of sources in a sky model is available from the
    :meth:`num_sources <oskar.Sky.num_sources>` property.
    A copy of the sky model data can be returned as a numpy array using the
    :meth:`to_array() <oskar.Sky.to_array()>` method,
    and written as a text file using the
    :meth:`save() <oskar.Sky.save()>` method.

    Note that the sky model used by OSKAR exists independently from the
    simulated observation parameters: make sure the phase centre is set
    appropriately when running simulations, otherwise you may not see what
    you were expecting!

    Some examples of setting up a sky model from Python are shown below.

    Examples:

        To load a sky model text file ``my_sky.txt``:

        >>> sky = oskar.Sky.load('my_sky.txt')

        To create a sky model from a numpy array:

        >>> # Specifying only RA, Dec and Stokes I (other columns default to 0).
        >>> data = numpy.array([[20.0, -30.0, 1],
                                [20.0, -30.5, 3],
                                [20.5, -30.5, 3]])
        >>> sky = oskar.Sky.from_array(data)
        >>> print(sky.num_sources)
        3

        To create a sky model from columns in a FITS binary table
        ``GLEAM_EGC.fits``:

        >>> from astropy.io import fits
        >>> hdulist = fits.open('GLEAM_EGC.fits')
        >>> cols = hdulist[1].data[0].array
        >>> data = numpy.column_stack(
                (cols['RAJ2000'], cols['DEJ2000'], cols['peak_flux_wide']))
        >>> sky = oskar.Sky.from_array(data)
        >>> print(sky.num_sources)
        307455

        To create an all-sky model containing 100000 sources with
        fluxes between 1 mJy and 100 mJy, and a power law index for
        the luminosity function of -2:

        >>> sky = oskar.Sky.generate_random_power_law(100000, 1e-3, 100e-3, -2)
        >>> print(sky.num_sources)
        100000

        To filter the sky model to contain sources only
        between 5 degrees and 15 degrees from the point
        (RA, Dec) = (0, 80) degrees:

        >>> # (continued from previous section)
        >>> ra0 = 0
        >>> dec0 = 80
        >>> sky.filter_by_radius(5, 15, ra0, dec0)
        >>> print(sky.num_sources)
        1521

        To plot the sky model using matplotlib:

        >>> # (continued from previous section)
        >>> import matplotlib.pyplot as plt
        >>> data = sky.to_array()  # Get sky model data as numpy array.
        >>> ra = numpy.radians(data[:, 0] - ra0)
        >>> dec = numpy.radians(data[:, 1])
        >>> log_flux = numpy.log10(data[:, 2])
        >>> x = numpy.cos(dec) * numpy.sin(ra)
        >>> y = numpy.cos(numpy.radians(dec0)) * numpy.sin(dec) - \\
                    numpy.sin(numpy.radians(dec0)) * numpy.cos(dec) * numpy.cos(ra)
        >>> sc = plt.scatter(x, y, s=5, c=log_flux, cmap='plasma',
                    vmin=numpy.min(log_flux), vmax=numpy.max(log_flux))
        >>> plt.axis('equal')
        >>> plt.xlabel('x direction cosine')
        >>> plt.ylabel('y direction cosine')
        >>> plt.colorbar(sc, label='Log10(Stokes I flux [Jy])')
        >>> plt.show()

        .. figure:: example_filtered_sky.png
           :width: 640px
           :align: center
           :height: 480px
           :alt: An example filtered sky model, plotted using matplotlib

           An example filtered sky model, plotted using matplotlib

    """

    def __init__(self, precision=None, settings=None):
        """Creates an OSKAR sky model.

        Args:
            precision (Optional[str]):
                Either 'double' or 'single' to specify the numerical
                precision of the sky model. Default 'double'.
            settings (Optional[oskar.SettingsTree]):
                Optional settings to use to set up the sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = None
        if precision is not None and settings is not None:
            raise RuntimeError("Specify either precision or all settings.")
        if precision is None:
            precision = 'double'  # Set default.
        if settings is not None:
            sky = settings.to_sky()
            self._capsule = sky.capsule
            self._settings = settings
        self._precision = precision

    def append(self, other):
        """Appends data from another sky model.

        Args:
            other (oskar.Sky): Another sky model.
        """
        self.capsule_ensure()
        _sky_lib.append(self._capsule, other.capsule)

    def append_sources(self, ra_deg, dec_deg, I, Q=None, U=None, V=None,
                       ref_freq_hz=None, spectral_index=None,
                       rotation_measure=None, major_axis_arcsec=None,
                       minor_axis_arcsec=None, position_angle_deg=None):
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
        self.capsule_ensure()
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
        _sky_lib.append_sources(
            self._capsule, numpy.radians(ra_deg), numpy.radians(dec_deg),
            I, Q, U, V, ref_freq_hz, spectral_index, rotation_measure,
            numpy.radians(major_axis_arcsec / 3600.0),
            numpy.radians(minor_axis_arcsec / 3600.0),
            numpy.radians(position_angle_deg))

    def append_file(self, filename):
        """Appends data to the sky model from a text file.

        Args:
            filename (str): Name of file to load.
        """
        self.capsule_ensure()
        _sky_lib.append_file(self._capsule, filename)

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            self._capsule = _sky_lib.create(self._precision)

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _sky_lib.capsule_name(new_capsule) == 'oskar_Sky':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_Sky.")

    def create_copy(self):
        """Creates a copy of the sky model.

        Returns:
            oskar.Sky: A copy of the sky model.
        """
        self.capsule_ensure()
        t = Sky()
        t.capsule = _sky_lib.create_copy(self._capsule)
        return t

    def filter_by_flux(self, min_flux_jy, max_flux_jy):
        """Filters the sky model according to Stokes-I flux.

        Sources with flux values outside the range are deleted.

        Args:
            min_flux_jy (float): Minimum allowed flux, in Jy.
            max_flux_jy (float): Maximum allowed flux, in Jy.
        """
        self.capsule_ensure()
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
        self.capsule_ensure()
        _sky_lib.filter_by_radius(
            self._capsule, math.radians(inner_radius_deg),
            math.radians(outer_radius_deg),
            math.radians(ra0_deg), math.radians(dec0_deg))

    @classmethod
    def from_array(cls, array, precision='double'):
        """Creates a new sky model from a 2D numpy array.

        The format of the array is the same as that in sky model text files.
        Each column specifies a different source parameter, and each row
        specifies data for a different source.

        The array could be created, for example, like:
        array = numpy.zeros((num_sources, num_parameters))

        There must be at least 3 columns present (RA, Dec, Stokes I).
        Parameters for missing columns will take default values.

        If the array is 1-dimensional, it will be treated as specifying
        parameters only for a single source.

        Args:
            array (float, array-like): Input array.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the sky model.

        Returns:
            oskar.Sky: The new sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = Sky()
        t.capsule = _sky_lib.from_array(array, precision)
        return t

    @classmethod
    def from_fits_file(cls, filename, min_peak_fraction=0.0, min_abs_val=0.0,
                       default_map_units='K', override_units=False,
                       frequency_hz=0.0, spectral_index=-0.7,
                       precision='double'):
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
                the numerical precision of the sky model.

        Returns:
            oskar.Sky: The new sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = Sky()
        t.capsule = _sky_lib.from_fits_file(
            filename, min_peak_fraction, min_abs_val, default_map_units,
            override_units, frequency_hz, spectral_index, precision)
        return t

    @classmethod
    def generate_grid(cls, ra0_deg, dec0_deg, side_length, fov_deg,
                      mean_flux_jy=1.0, std_flux_jy=0.0, seed=1,
                      precision='double'):
        """Generates a grid of sources and returns it as a new sky model.

        Args:
            ra0_deg (float):      Right Ascension of grid centre, in degrees.
            dec0_deg (float):     Declination of grid centre, in degrees.
            side_length (int):    Side length of generated grid.
            fov_deg (float):      Grid field-of-view, in degrees.
            mean_flux_jy (float): Mean Stokes-I source flux, in Jy.
            std_flux_jy (float):  Standard deviation Stokes-I flux, in Jy.
            seed (int):           Random generator seed.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the sky model.

        Returns:
            oskar.Sky: The new sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = Sky()
        t.capsule = _sky_lib.generate_grid(
            ra0_deg, dec0_deg, side_length, fov_deg, mean_flux_jy,
            std_flux_jy, seed, precision)
        return t

    @classmethod
    def generate_random_power_law(cls, num_sources, min_flux_jy, max_flux_jy,
                                  power_law_index, seed=1, precision='double'):
        """Generates sources scattered randomly over the celestial sphere.

        Args:
            num_sources (int):       The number of sources to generate.
            min_flux_jy (float):     Minimum Stokes-I source flux, in Jy.
            max_flux_jy (float):     Maximum Stokes-I source flux, in Jy.
            power_law_index (float): Power law index/exponent.
            seed (int):              Random generator seed.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the sky model.

        Returns:
            oskar.Sky: The new sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = Sky()
        t.capsule = _sky_lib.generate_random_power_law(
            num_sources, min_flux_jy, max_flux_jy, power_law_index, seed,
            precision)
        return t

    @classmethod
    def load(cls, filename, precision='double'):
        """Loads data from a text file and returns it as a new sky model.

        Args:
            filename (str): Name of file to load.
            precision (Optional[str]): Either 'double' or 'single' to specify
                the numerical precision of the sky model.

        Returns:
            oskar.Sky: The new sky model.
        """
        if _sky_lib is None:
            raise RuntimeError("OSKAR library not found.")
        t = Sky()
        t.capsule = _sky_lib.load(filename, precision)
        return t

    def save(self, filename):
        """Saves data to a sky model text file.

        Args:
            filename (str): Name of file to write.
        """
        self.capsule_ensure()
        _sky_lib.save(self._capsule, filename)

    def to_array(self):
        """Returns a copy of the sky model as a 2D numpy array.

        Returns:
            numpy.ndarray: A copy of the sky model.
        """
        self.capsule_ensure()
        array = _sky_lib.to_array(self._capsule)
        array[:, 0] *= (180.0 / math.pi)
        array[:, 1] *= (180.0 / math.pi)
        array[:, 9] *= (180.0 / math.pi) * 3600.0
        array[:, 10] *= (180.0 / math.pi) * 3600.0
        array[:, 11] *= (180.0 / math.pi)
        return array

    def to_ds9_regions(self, filename, colour='green', width=1):
        """Writes a ds9 region file from coordinates in the sky model.

        Args:
            filename (str): Name of file to write.
            colour (Optional[str]): Name of region colour. Default 'green'.
            width (Optional[int]): Line width. Default 1.
        """
        self.capsule_ensure()
        array = _sky_lib.to_array(self._capsule)
        array[:, 0] *= (180.0 / math.pi)
        array[:, 1] *= (180.0 / math.pi)
        with open(filename, "w") as file:
            file.write('global color=%s dashlist=8 3 width=%d '
                       'font="helvetica 10 normal roman" select=1 highlite=1 '
                       'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 '
                       'source=1\nfk5\n' % (colour, width))
            for i in range(0, len(array)):
                file.write('point(%10.6f, %10.6f) # point=x\n' %
                           (array[i, 0], array[i, 1]))

    # Properties
    capsule = property(capsule_get, capsule_set)

    @property
    def num_sources(self):
        """Returns the number of sources in the sky model.

        Type
            int
        """
        self.capsule_ensure()
        return _sky_lib.num_sources(self._capsule)
