.. |br| raw:: html

   <br />

.. _sky-model:

*********
Sky Model
*********

This section describes the sky model file formats recognised by OSKAR 2.x.

Sky model files contain a simple catalogue to describe the source
properties for a set of point sources or Gaussian sources.

Since OSKAR 2.12, it is possible to load and save a large subset of
sky model parameters supported by LOFAR software, including BBS, DP3
and WSClean.
This is the "named-column" format described below.


Sky Model File (named-column format)
====================================

This flexible format uses a single-line format string in the file
header, which defines which columns are present, and in which order.
Each row corresponds to one source, and columns describe the source
parameters.
The format string is described in some detail on the
`LOFAR Wiki page <https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#format_string>`_.

Format strings supported by OSKAR include having the "Format =" specifier
either at the start or the end of a line, with field types optionally
enclosed in brackets, spelled in mixed case, or embedded within a comment,
and space- and/or comma-separated.
The only requirement is that "Format" must appear at the start or end of a
line (neglecting comment characters and whitespace), and have an equals '='
character either before or after the word.
So all these format strings, and variations thereof, would be accepted by OSKAR:

* ``Format = RA, Dec, I``
* ``# format = RA Dec I``
* ``Format= (Ra, Dec, I, Q, U, V)``
* ``# (RA,Dec,I,Q,U) = format``

The field types in the format string are reserved names to specify the type
of data in each column of the text file.
Field type names supported by OSKAR are case-insensitive, and include:

.. csv-table::
   :header: "Field type", "Unit", "Description"
   :widths: 22, 12, 66

   **Ra**, angle, "Right Ascension, in decimal degrees or radians (default);
   |br| or sexagesimal hours, minutes and seconds.
   |br| See note below."
   **Dec**, angle, "Declination, in decimal degrees or radians (default);
   |br| or sexagesimal degrees, minutes and seconds.
   |br| See note below."
   **RaD**, angle, "Right Ascension, in decimal degrees (default) or radians;
   |br| or sexagesimal hours, minutes and seconds. |br|
   Use instead of **Ra** if required. See note below."
   **DecD**, angle, "Declination, in decimal degrees (default) or radians;
   |br| or sexagesimal degrees, minutes and seconds. |br|
   Use instead of **Dec** if required. See note below."
   **I** |br| or **StokesI**, Jy, "Stokes I flux (at reference frequency)."
   **Q** |br| or **StokesQ**, Jy, "Optional Stokes Q flux."
   **U** |br| or **StokesU**, Jy, "Optional Stokes U flux."
   **V** |br| or **StokesV**, Jy, "Optional Stokes V flux."
   **ReferenceFrequency**, Hz, "Optional reference frequency for source fluxes."
   **SpectralIndex**, N/A, "Optional spectral index polynomial; can be a
   multi-valued |br| vector, with a list of values enclosed in brackets;
   up to 8 terms |br| are supported. See :ref:`spectral-profiles` below."
   **LogarithmicSI**, boolean, "Optional boolean flag: If true, spectral
   indices are logarithmic, |br| otherwise linear; see the
   `LOFAR Wiki page on LogarithmicSI <https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#logarithmic_spectral_index>`_.
   |br| Default true if omitted."
   **MajorAxis**, arcsec, "Optional Gaussian source FWHM major axis."
   **MinorAxis**, arcsec, "Optional Gaussian source FWHM minor axis."
   **Orientation** |br| or **PositionAngle**, deg, "Optional position angle
   of Gaussian major axis."
   **RotationMeasure**, rad/m^2, "Optional source rotation measure."
   **PolarizationAngle** |br| or **PolarisationAngle**, deg, "Optional
   source polarisation angle; used if Q and U are |br| omitted, or when
   a rotation measure is set."
   **PolarizedFraction** |br| or **PolarisedFraction**, N/A, "Optional
   fraction of linear polarisation, used if Q and U are |br| omitted, or when
   a rotation measure is set."
   **ReferenceWavelength**, metres, "Optional reference wavelength,
   used with the rotation |br| measure parameter. If omitted, it will be
   calculated based on |br| the reference frequency."
   **SpectralCurvature**, N/A, "Optional spectral curvature term described in
   |br| `Callingham et. al. (2017) <https://iopscience.iop.org/article/10.3847/1538-4357/836/2/174/pdf>`_,
   equation 2, where this value is |br| interpreted as the parameter 'q'.
   If non-zero, only the first |br| **SpectralIndex** value will be used,
   and any others will be |br| ignored. See :ref:`spectral-curvature` below."
   **LineWidth**, Hz, "Optional line width in Hz, if this is a spectral line
   source. |br| If the line width is greater than 0, then spectral index
   |br| values will be ignored, and the Stokes I flux will be calculated
   |br| using a Gaussian profile centred on the reference frequency.
   |br| See :ref:`spectral-line-profile` below."

.. tip::
   - Columns may appear in any order, and optional columns may be omitted
     entirely.
   - Unknown column types will be ignored when the file is loaded - note that
     this includes columns **Name** and **Type** used by LOFAR software.
   - Gaussian sources are specified using non-zero values in both
     **MajorAxis** and **MinorAxis** columns. Gaussian sources also need to
     specify an **Orientation** (or **PositionAngle**), even if it is zero.

.. warning::
   If the **ReferenceFrequency** is omitted or set to zero, the source flux
   cannot be re-calculated as a function of frequency, even if spectral index
   and/or rotation measure values are specified.
   In this case, the source flux values will be the same at every frequency.

.. note::
   The coordinate values used in the (**RA**, **Dec**) columns may have a
   suffix added to define the unit, either "rad" or "deg" respectively
   for radians or degrees.
   For consistency, if the unit is omitted, radians is assumed for
   both the "**Ra**" and "**Dec**" columns, and degrees is assumed if the
   column names are instead "**RaD**" or "**DecD**".

.. note::
   Sources of different spectral types can be combined within the same sky
   model, if the relevant columns are specified. If all the columns are
   present, the priority order is:

   1. A spectral line profile will be used for the source if **LineWidth** is
      greater than zero;
   2. Otherwise, a spectral curvature model for the source will be used
      if **SpectralCurvature** is non-zero;
   3. Otherwise, a logarithmic or linear spectral index polynomial
      will be used.

.. note::
   If a **RotationMeasure** is defined, it will be used along with the
   **PolarizationAngle**, **PolarizedFraction** and **ReferenceWavelength**
   parameters according to the logic described in the
   `BBS chapter of the LOFAR Imaging Cookbook <https://support.astron.nl/LOFARImagingCookbook/bbs.html#rotation-measure>`_

In addition, all columns can take a default value, which is specified
in the format string header inside quotes, after an '=' character.
The default will be used for all sources which do not explicitly set that
parameter value.
If a default is set, there must be no spaces either before or after the '='.

For example, to specify a common reference frequency for all sources in the
sky model, the following format string could be used:

``Format = RaD, DecD, I, ReferenceFrequency='143e6', MajorAxis, MinorAxis``

The fields can be space-separated and/or comma-separated. Characters
appearing after a hash (``#``) symbol are treated as comments and will be
ignored. Empty lines are also ignored.

Example
-------

.. code-block:: text

   # Number of sources: 3
   # (Name, Type, Ra, Dec, I, ReferenceFrequency='100e6', SpectralIndex, Q, U, MajorAxis, MinorAxis, Orientation, V) = format
   s1,POINT,20deg,-30deg,1,,-0.7,0,0,,,,0
   s2,GAUSSIAN,20deg,-30.5deg,3,,-0.7,2,2,600,50,45,0
   s3,GAUSSIAN,20.5deg,-30.5deg,3,,-0.7,0,0,700,10,-10,2

.. raw:: latex

    \clearpage

Sky Model File (fixed format)
=============================

The original fixed-format sky model file used by OSKAR holds a simple
text-based table, where each row corresponds to one source, and columns
describe the source parameters.
The column order is implicit and cannot be changed.
Most parameters are optional, and will be set to a default value of zero if not
specified.
The defaults in this format cannot be changed.

Although this format is less flexible and supports fewer features than the
named-column format described above, it can be easier to parse, and is
therefore still supported for simple sky models.
In particular, it can be loaded into Python very straightforwardly
using ``numpy.loadtxt()``.

.. note::
   When the file is read, parameters are assigned according to their column
   position. In order to specify an optional parameter, all columns up to the
   designated column must be specified.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Parameter", "Unit", "Comment"
   :widths: 10, 22, 12, 56

   1, "Right Ascension", "deg", "Required. Currently interpreted as the apparent |br| rather than mean (J2000) Right Ascension."
   2, "Declination", "deg", "Required. Currently interpreted as the apparent |br| rather than mean (J2000) Declination."
   3, "Stokes I flux", "Jy", "Required."
   4, "Stokes Q flux", "Jy", "Optional (default 0)."
   5, "Stokes U flux", "Jy", "Optional (default 0)."
   6, "Stokes V flux", "Jy", "Optional (default 0)."
   7, "Reference frequency", "Hz", "Optional (default 0). |br| Frequency at which flux densities are given."
   8, "Spectral index", "N/A", "Optional (default 0)."
   9, "Rotation measure", "rad / m^2", "Optional (default 0)."
   10, "Major axis FWHM", "arcsec", "Optional (default 0)."
   11, "Minor axis FWHM", "arcsec", "Optional (default 0)."
   12, "Position angle", "deg", "Optional (default 0). East of North."

.. note::
   In order for a source to be recognised as a Gaussian, all three of the
   major axis FWHM, minor axis FWHM and position angle parameters must be
   specified.

.. note::
   The rotation measure column was added for OSKAR 2.3.0. To provide backwards
   compatibility with older sky model files containing extended sources, a
   check is made on the number of columns on each line, and source data is
   loaded according to the following rules:

   1. Lines containing between 3 and 9 columns will set all parameters
      up to and including the rotation measure. Any missing parameters will
      be set to defaults.
   2. Lines containing 11 columns set the first 8 parameters and the
      Gaussian source data (this is the old file format). The rotation
      measure will be set to zero.
   3. Lines containing 12 columns set all parameters.
   4. Lines containing 10, 13, or more columns will raise an error.

The fields can be space-separated and/or comma-separated. Characters
appearing after a hash (``#``) symbol are treated as comments and will be
ignored. Empty lines are also ignored.


Example
-------

The following is an example sky file describing three sources, making use of
a number of comment lines.

.. code-block:: text

   #
   # Required columns:
   # =================
   # RA(deg), Dec(deg), I(Jy)
   #
   # Optional columns:
   # =================
   # Q(Jy), U(Jy), V(Jy), freq0(Hz), spectral index, rotation measure,
   #           FWHM major (arcsec), FWHM minor (arcsec), position angle (deg)
   #
   #
   # Two fully-specified sources
   0.0 70.0 1.1 0.0 0.0 0.0 100e6 -0.7 0.0 200.0 150.0  23.0
   0.0 71.2 2.3 1.0 0.0 0.0 100e6 -0.7 0.0  90.0  40.0 -10.0

   # A source where only Stokes I is defined (other columns take default values)
   0.1 69.8 1.0

.. raw:: latex

    \clearpage

.. _spectral-profiles:

Spectral Profiles
=================

The spectral index parameters are used as described on the
`LOFAR Wiki page detailing logarithmic and linear spectral indices <https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#logarithmic_spectral_index>`_.

Logarithmic Polynomial
----------------------
The default logarithmic spectral indices
(:math:`\alpha_0, \alpha_1, \alpha_2 \cdots`) are used to scale the
flux :math:`S_0` given at the reference frequency :math:`\nu_0` to
another frequency :math:`\nu` as follows:

.. math:: S_{\nu} = S_0 \left( \frac{\nu}{\nu_0} \right)^{\alpha_0 + \alpha_1 \log_{10}\left( \frac{\nu}{\nu_0} \right) + \alpha_2 \log_{10}\left( \frac{\nu}{\nu_0} \right)^2 + \cdots }

If using only a single spectral index value :math:`\alpha_0`, this reduces
to the usual expression of
:math:`S_{\nu} = S_0 \left(\nu / \nu_0 \right)^{\alpha_0}`.

Linear Polynomial
-----------------
The linear spectral indices used by WSClean are also supported, and used if
**LogarithmicSI** is false.
In this case, the flux :math:`S_0` given at the reference frequency
:math:`\nu_0` scales to another frequency :math:`\nu` as follows:

.. math:: S_{\nu} = S_0 + \alpha_0 \left( \frac{\nu}{\nu_0} - 1 \right) + \alpha_1 \left( \frac{\nu}{\nu_0} - 1 \right)^2 + \alpha_2 \left( \frac{\nu}{\nu_0} - 1 \right)^3 + \cdots

.. _spectral-curvature:

Spectral Curvature
------------------
If specified, and not 0, the **SpectralCurvature** parameter (:math:`q`, below)
is used with the first spectral index value (:math:`\alpha_0`) to scale the
flux :math:`S_0` given at the reference frequency :math:`\nu_0` to another
frequency :math:`\nu` as follows:

.. math:: S_{\nu} = S_0 \left( \frac{\nu}{\nu_0} \right)^{\alpha_0} \exp\left( q \ln\left( \frac{\nu}{\nu_0} \right)^2 \right)

.. _spectral-line-profile:

Spectral Line Profile
---------------------
If a **LineWidth** parameter (:math:`\sigma`, below) is specified and greater
than 0, then the source will be treated as a spectral line source with a
Gaussian profile centred at the reference frequency :math:`\nu_0`, with a peak
flux given by the **StokesI** parameter.
Any spectral index values will be ignored in this case. The flux at a frequency
:math:`\nu` will be calculated as follows:

.. math:: S_{\nu} = S_0 \exp\left(- \frac{(\nu - \nu_0)^2}{2 \sigma^2}\right)

.. raw:: latex

    \clearpage

Gaussian Sources
================

Two-dimensional elliptical Gaussian sources are specified by the length of
their major and minor axes on the sky in terms of their full width at half
maximum (FWHM) and the position angle of the major axis :math:`\theta`,
defined as the angle East of North.

.. figure:: gaussian.png
   :width: 7cm
   :align: center
   :alt: Gaussian source definition

These three parameters define an elliptical Gaussian :math:`f(x,y)`, given by
the equation

.. math:: f(x,y)=\exp\left\{-(ax^2 + 2bxy + cy^2) \right\}

where

.. math::

   a &= \frac{\cos^2 \theta}{2\sigma_x^2} + \frac{\sin^2 \theta}{2\sigma_y^2} \\
   b &= -\frac{\sin2\theta}{4\sigma_x^2} + \frac{\cos2\theta}{4\sigma_y^2} \\
   c &= \frac{sin^2 \theta}{2\sigma_x^2} + \frac{\cos^2 \theta}{2\sigma_y^2},

and :math:`\sigma_x`  and :math:`\sigma_y` are related to the minor and major
FWHM respectively, according to

.. math:: \sigma = \frac{\rm FWHM}{ 2 \sqrt{2 \ln(2)}} .

OSKAR simulates Gaussian sources by multiplying the amplitude
response of the source on each baseline by the Gaussian response of the source
in the :math:`(u,v)` plane. This is possible in the limit where a Gaussian source
differs from a point source in its Fourier :math:`(u,v)` plane response only,
and assumes that any variation of Jones matrices across the extent of the
source can be ignored (e.g. a small taper due to the station beam changing
across the source).

The Fourier response of an elliptical Gaussian source is another elliptical
Gaussian whose width is defined with respect to the width in the sky as

.. math:: \sigma_{uv} = \frac{1}{2 \pi \sigma_{\rm sky}} .

The required modification of the :math:`(u, v)` plane amplitude response of
each point source therefore takes the simple analytical form
:math:`V_{\rm extended} = f(u,v) \, V_{\rm point}`,
where :math:`f(u,v)` is the equation for an elliptical Gaussian (defined above as
:math:`f(x,y)`) evaluated in the :math:`(u,v)` plane according to the FWHM and
position angle of the source.
