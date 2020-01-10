
Sky Model
=========

This class can be used to create an OSKAR sky model from Python.
The sky model can be thought of as simply a table of source data, where
each row of the table contains parameters for a single source.
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

Note that the sky model used by OSKAR exists independently from the simulated
observation parameters: make sure the phase centre is set appropriately
when running simulations, otherwise you may not see what you were expecting!

Some examples of setting up a sky model from Python are shown below.

Example Usage
-------------

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

To create a sky model from columns in a FITS binary table ``GLEAM_EGC.fits``:

>>> from astropy.io import fits
>>> hdulist = fits.open('GLEAM_EGC.fits')
>>> cols = hdulist[1].data[0].array
>>> data = numpy.column_stack(
        (cols['RAJ2000'], cols['DEJ2000'], cols['peak_flux_wide']))
>>> sky = oskar.Sky.from_array(data)
>>> print(sky.num_sources)
307455

To create an all-sky model containing 100000 sources with fluxes
between 1 mJy and 100 mJy, and a power law index for the
luminosity function of -2:

>>> sky = oskar.Sky.generate_random_power_law(100000, 1e-3, 100e-3, -2)
>>> print(sky.num_sources)
100000

To filter the sky model to contain sources only
between 5 degrees and 15 degrees from the point (RA, Dec) = (0, 80) degrees:

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
>>> y = numpy.cos(numpy.radians(dec0)) * numpy.sin(dec) - \
            numpy.sin(numpy.radians(dec0)) * numpy.cos(dec) * numpy.cos(ra)
>>> sc = plt.scatter(x, y, s=5, c=log_flux, cmap='plasma',
            vmin=numpy.min(log_flux), vmax=numpy.max(log_flux))
>>> plt.axis('equal')
>>> plt.xlabel('x direction cosine')
>>> plt.ylabel('y direction cosine')
>>> plt.colorbar(sc, label='Log10(Stokes I flux [Jy])')
>>> plt.show()

.. image:: example_filtered_sky.png
   :width: 640px
   :align: center
   :height: 480px
   :alt: An example filtered sky model, plotted using matplotlib


Class Methods
-------------

.. autoclass:: oskar.Sky
   :members:
   :special-members: __init__
   :exclude-members: capsule, capsule_ensure, capsule_get, capsule_set
