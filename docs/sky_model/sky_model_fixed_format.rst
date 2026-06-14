.. |br| raw:: html

   <br />

.. _sky-model-file-fixed-format:

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
:ref:`named-column format <sky-model-file-named-column-format>`, it can be
easier to parse, and is therefore still supported for simple sky models.
In particular, it can be loaded into Python very straightforwardly
using ``numpy.loadtxt()``.

.. note::
   When the file is read, parameters are assigned according to their column
   position. In order to specify an optional parameter, all columns up to the
   designated column must be specified.

Warning!
--------

This appears to be a common source of confusion, so please take note:

.. warning::
   **There is no machine-readable file header in this format.**

   **Do not rely on any comments being present in fixed-format OSKAR sky model files.**
   Any comments are intended to be human-readable only.
   Source parameters in the old fixed-format text file are assigned
   based **only** on their column index positions.

   The fixed-format OSKAR sky model file cannot support new features, and is
   provided mainly for backwards-compatibility.
   Use the self-describing
   :ref:`named-column format <sky-model-file-named-column-format>`
   in preference.

In order, the parameter columns in the old fixed-format sky model file are:

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
   9, "Rotation measure", "rad / m\ :sup:`2`", "Optional (default 0)."
   10, "Major axis FWHM", "arcsec", "Optional (default 0)."
   11, "Minor axis FWHM", "arcsec", "Optional (default 0)."
   12, "Position angle", "deg", "Optional (default 0). East of North."

.. note::
   In order for a source to be recognised as a Gaussian, all three of the
   major axis FWHM, minor axis FWHM and position angle parameters must be
   specified.

.. note::
   The rotation measure column was added for OSKAR 2.3.0. To provide backwards
   compatibility with even older sky model files containing extended sources, a
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
