.. _telescope_model:

***************
Telescope Model
***************

This document describes the format of the telescope model used by OSKAR
versions >= 2.7. The telescope model includes a description of the position of
each station in the interferometer, the configuration of each station, and
optional custom antenna (element) patterns.

Directory Structure
===================

A telescope model is defined using a directory structure. The name of the
top-level directory is arbitrary, but it must contain a special file to
specify the telescope centre position, a special file to specify the position
of each station, and a set of sub-directories (again, with arbitrary names),
one for every station. Each of these sub-directories contains one or more
special files to specify the configuration of that station.

The following section shows the names of special files allowed at each level
in the directory structure.

Station directories may themselves contain further directories to describe
the configuration of sub-stations (or "tiles") that will be beamformed
hierarchically. For a two-level hierarchical station, the top station level
describes the layout of the tiles with respect to the station centre, and
files in the sub-directories describe the layout of each tile with respect
to its own centre.

The **alphabetical** order of the station directories corresponds to the order
in which the station coordinates and element data appear in the layout and
configuration files, if there is more than one station type.
(**Note that leading zeros must be used in directory names when necessary.**)
If instead all stations are the same, then only a single (sub-)station directory
should be present.

Since OSKAR 2.8, it is possible to specify the type of each top-level station
using a file to map station type index to station index, which is useful if
there are fewer station types than there are stations in the telescope.
An example of how this could look is shown at the end of this section.

Example Telescope Model (single level beamforming; general case)
----------------------------------------------------------------
An example telescope model directory might contain the following:

- my_telescope_model/

  - station001/

    - [files describing the configuration of station 1]

  - station002/

    - [files describing the configuration of station 2]

  - station003/

    - [files describing the configuration of station 3]

  - [other station directories]

  - [file describing the layout of stations in the interferometer]

  - [file describing the centre position of the interferometer]


Example Telescope Model (single level beamforming; all stations identical)
--------------------------------------------------------------------------
For stations that are all identical, it is sufficient to specify only one
top-level station directory as follows:

- my_telescope_model/

  - station/

    - [files describing the configuration of all stations]

  - [file describing the layout of stations in the interferometer]

  - [file describing the centre position of the interferometer]


Example Telescope Model (two-level beamforming; general case)
-------------------------------------------------------------
An example telescope model directory where stations are composed of tiles
could look like this:

- my_telescope_model/

  - station001/

    - tile001/

      - [files describing the configuration of tile 1 in station 1]

    - tile002/

      - [files describing the configuration of tile 2 in station 1]

    - [other tile directories]

    - [file describing the layout of tiles in station 1]

  - station002/

    - tile001/

      - [files describing the configuration of tile 1 in station 2]

    - tile002/

      - [files describing the configuration of tile 2 in station 2]

    - [other tile directories]

    - [file describing the layout of tiles in station 2]

  - [other station directories]

  - [file describing the layout of stations in the interferometer]

  - [file describing the centre position of the interferometer]


Example Telescope Model (two-level beamforming; all tiles and stations identical)
---------------------------------------------------------------------------------
For hierarchical stations that are all identical, and made of identical tiles,
it is sufficient to specify only one station directory at each level:

- my_telescope_model/

  - station/

    - tile/

      - [files describing the configuration of each tile]

    - [file describing the layout of tiles in the station]

  - [file describing the layout of stations in the interferometer]

  - [file describing the centre position of the interferometer]


Example Telescope Model (single level beamforming; use of station type map)
---------------------------------------------------------------------------
If there are only a small number of station types in the telescope model, then
use the station directories to specify the station types, and add another file
in the root of the telescope model to specify the type of each station:

- my_telescope_model/

  - station-type000/

    - [files describing the configuration of station type 0]

  - station-type001/

    - [files describing the configuration of station type 1]

  - [file describing the layout of stations in the interferometer]

  - [file describing the centre position of the interferometer]

  - [file describing the mapping between station type and station index]

Note that station type indices are zero-based, and must appear as such in the
mapping file (see :ref:`telescope_station_type_mapping`).

.. raw:: latex

    \clearpage


Special Files
=============
This section shows the names of files that may be present in the various
directories of the telescope model.

Fields in text files can be space-separated and/or comma-separated.
Characters appearing after a hash (``#``) symbol are treated as comments,
and any further characters on that line are ignored.
Empty lines are also ignored.

* ``position.txt``

  Centre reference position of telescope array.

  * **See** :ref:`telescope_position`

  * **Required:** Yes.

  * **Allowed locations:** Telescope model root directory.


* ``layout.txt``

  The layout (in horizontal East-North-Up coordinates) of stations or elements
  within stations.

  * **See** :ref:`telescope_layout_files`

  * **Required:** Yes (but see below).

  * **Allowed locations:** All directories.


* ``layout_ecef.txt``

  The layout of stations in Earth-centred-Earth-fixed coordinates.
  Can be used instead of ``layout.txt`` or ``layout_wgs84.txt`` at
  top-level only, if required.

  * **See** :ref:`telescope_layout_ecef`

  * **Required:** No, unless layout.txt and layout_wgs84.txt are omitted.

  * **Allowed locations:** Telescope model root directory.


* ``layout_wgs84.txt``

  The layout of stations in WGS84 (longitude, latitude, altitude) coordinates.
  Can be used instead of ``layout.txt`` or ``layout_ecef.txt`` at
  top-level only, if required.

  * **See** :ref:`telescope_layout_wgs84`

  * **Required:** No, unless layout.txt and layout_ecef.txt are omitted.

  * **Allowed locations:** Telescope model root directory.


* ``station_type_map.txt``

  If using station directories to specify only a limited number of
  station types, this file contains the mapping between the station ID and
  station type.

  * **See** :ref:`telescope_station_type_mapping`

  * **Required:** No, unless the number of station folders is greater than 1
    and less than the number of stations.

  * **Allowed locations:** Telescope model root directory.


.. raw:: latex

    \clearpage


Element Data
------------
* ``element_types.txt``

  Type index of each element in the station.

  * **See** :ref:`telescope_element_types`

  * **Required:** No.

  * **Allowed locations:** Station directory.


* ``gain_phase.txt``

  Per-element gain and phase offsets and errors.

  * **See** :ref:`telescope_element_gain_phase_errors`

  * **Required:** No.

  * **Allowed locations:** Station directory.


* ``cable_length_error.txt``

  Per-element cable length errors.

  * **See** :ref:`telescope_element_cable_length_errors`

  * **Required:** No.

  * **Allowed locations:** Station directory.


* ``apodisation.txt`` | ``apodization.txt``

  Per-element complex apodisation weight.

  * **See** :ref:`telescope_element_apodisation`

  * **Required:** No.

  * **Allowed locations:** Station directory.


* ``feed_angle.txt`` | ``feed_angle_x.txt`` | ``feed_angle_y.txt``

  Per-element and per-polarisation feed angles.

  * **See** :ref:`telescope_element_feed_angle`

  * **Required:** No.

  * **Allowed locations:** Station directory.


Element Type Data
-----------------
* ``element_pattern_fit_*.bin``

  Fitted element X-or Y-dipole responses for the station, as a function of frequency.

  * **See** :ref:`telescope_element_patterns_numerical`

  * **Required:** No.

  * **Allowed locations:** Any. (Inherited.)


* ``element_pattern_spherical_wave_*.txt``

  Fitted spherical wave element coefficient data, as a function of frequency.

  * **See** :ref:`telescope_element_patterns_spherical_wave`

  * **Required:** No.

  * **Allowed locations:** Any. (Inherited.)


.. * ``element_pattern_*.txt`` (Not implemented)

     Functional element X-or Y-dipole responses for the station.

     * **See** :ref:`telescope_element_patterns_functional`

     * **Required:** No.

     * **Allowed locations:** Any. (Inherited.)


.. raw:: latex

    \clearpage

Station Data
------------
.. * ``mount_type.txt`` (Not implemented)

    Mount type of station platform.

    * **See** :ref:`telescope_mount_type`

    * **Required:** No.

    * **Allowed locations:** Station directory.

* ``permitted_beams.txt``

  Permitted station beam directions.

  * **See** :ref:`telescope_permitted_beams`

  * **Required:** No.

  * **Allowed locations:** Station directory.


Gain Data
---------

* ``gain_model.h5``

  Externally-generated HDF5 file containing antenna or station gains, as
  a function of time, frequency, antenna/station and polarisation.
  This file may appear in any directory, but the size of the antenna or
  station dimension in each file must correspond to the number of antennas
  or stations in that directory.

  * **See** :ref:`telescope_gain_model`

  * **Required:** No.

  * **Allowed locations:** All directories.


HARP Data
---------

* ``*HARP*.h5``

  HDF5 file containing HARP station beam coefficient data.

  * **See** :ref:`telescope_harp`

  * **Required:** No.

  * **Allowed locations:** Station directory.


Noise Configuration Files
-------------------------
* ``noise_frequencies.txt``

  Frequencies for which noise values are defined.

  * **See** :ref:`telescope_noise`

  * **Required:** No, unless another noise file is present.

  * **Allowed locations:** Telescope model root directory. (Inherited.)


* ``rms.txt``

  Flux density RMS noise values, in Jy, as a function of frequency.

  * **See** :ref:`telescope_noise`

  * **Required:** No.

  * **Allowed locations:** Telescope model root directory, or top-level
    station directory. (Inherited.)


.. raw:: latex

    \clearpage


.. _telescope_position:

Position File
=============
The top-level "position.txt" file must specify the longitude and latitude
of the telescope origin. It must contain one line with two or three numbers:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 60, 25

   1, "WGS84 longitude, in degrees", "Required"
   2, "WGS84 latitude, in degrees", "Required"
   3, "Altitude, in metres", "Optional (default 0)"


.. _telescope_layout_files:

Layout Files
============
Layout files contain coordinates of stations or elements at (respectively)
the telescope or station level.

.. _telescope_layout:

Telescope Level
---------------
The top-level ``layout.txt`` file contains a table of ASCII text to represent
station positions relative to the centre of the interferometer array specified
in the :ref:`telescope_position`.
Each line contains up to six values, which correspond to positions
represented as horizontal :math:`(x, y, z)` coordinates in metres relative to a
local tangent (horizon) plane, where x is towards geographic east, y is towards
geographic north, and z is towards the local zenith.

Coordinate errors can also be specified using optional columns.
The first three columns are the "measured" positions, while
the "true" positions are obtained by adding the supplied offsets to the
measured values. Coordinates are given in metres.
In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 60, 25

   1, "Horizontal x (east) coordinate", "Required"
   2, "Horizontal y (north) coordinate", "Required"
   3, "Horizontal z (up) coordinate", "Optional (default 0)"
   4, "Horizontal x (east) coordinate error", "Optional (default 0)"
   5, "Horizontal y (north) coordinate error", "Optional (default 0)"
   6, "Horizontal z (up) coordinate error", "Optional (default 0)"

.. _telescope_layout_ecef:

Telescope Level Earth-centred Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many radio interferometers specify station positions in Earth-centred
coordinates. It is possible to do the same in OSKAR by using a file
named ``layout_ecef.txt`` instead of ``layout.txt`` in the top-level
telescope directory. Coordinates are given in metres.
In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 60, 25

   1, "ECEF x coordinate (towards longitude 0, latitude 0)", "Required"
   2, "ECEF y coordinate (towards the east)", "Required"
   3, "ECEF z coordinate (towards the north pole)", "Required"
   4, "ECEF x coordinate error", "Optional (default 0)"
   5, "ECEF y coordinate error", "Optional (default 0)"
   6, "ECEF z coordinate error", "Optional (default 0)"

.. raw:: latex

    \clearpage

.. _telescope_layout_wgs84:

Telescope Level WGS84 Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As a final option, it is possible to specify station positions as
WGS84 (longitude, latitude, altitude) values by using a file
named ``layout_wgs84.txt`` instead of ``layout.txt`` in the top-level
telescope directory.
In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 60, 25

   1, "WGS84 longitude, in degrees", "Required"
   2, "WGS84 latitude, in degrees", "Required"
   3, "Altitude, in metres", "Optional (default 0)"

.. _telescope_station_layout:

Station Level
-------------
In each station directory, there must be a ``layout.txt`` file to specify
the element position in horizontal :math:`(x, y, z)` coordinates relative to
the station centre, and (optionally) the :math:`(x, y, z)` position errors.

The format of the element layout file is the same as that used for the
telescope-level station coordinates in the horizon frame. It is not possible
to use Earth-centred or WGS84 coordinates to specify element locations within
a station.

.. The station layouts are defined with respect to the plane of
   the station platform. Use caution if the mount type of the station
   platform is not 'Fixed', as these coordinates will then not correspond
   to geographic directions.


.. _telescope_station_type_mapping:

Station Type Mapping
====================
For some telescope models, it may be convenient to only store station data
for a limited number of station types. In this case, the station directories
can be used to specify data for each type of station, and a further file
named ``station_type_map.txt`` at the top level of the telescope model is used
to assign each station to a station type.

If present, this file must contain a single column of integers, one row per
station, where the value of the integer corresponds to the
alphabetically-sorted index of each station directory in the telescope
model.

The order of the station indices in the station type map matches the
order of stations specified in the top-level station layout file.

If the station type map is missing, the station type map is set implicitly,
depending on the number of station directories in the telescope model:

* If there is only one station directory, all stations are assumed to be
  identical, and will be assigned a type of 0.
  (This is the same as in previous versions of OSKAR.)

* If the number of station directories matches the number of stations defined
  in the layout file, all stations are assumed to be different, and will be
  assigned a type of 0 to (num_stations - 1).

* Otherwise, an error will be reported if the number of station directories
  is different to the number of stations in the layout file **and** the
  station type map file is missing.

.. csv-table::
   :header: "Column", "Description"
   :widths: 15, 85

   1, "Integer index of station folder to use (per station; zero-based)."

.. raw:: latex

    \clearpage


.. _telescope_element_types:

Element Types
=============
In each station directory, there may be optionally an ``element_types.txt``
file to specify the type of each element in the station. This type index is
used in conjunction with element pattern data to select the correct file
of fitted coefficients.

If the element types file is omitted, all elements have an implicit type of 0.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description"
   :widths: 15, 85

   1, "Integer element type index (per element; zero-based)."


.. _telescope_element_gain_phase_errors:

Element Gain & Phase Error Files
================================
In each station directory, there may be optionally a ``gain_phase.txt`` file to
specify the per-element systematic and time-variable gain and phase errors.

Phases are given in degrees. In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 60, 25

   1, "Systematic gain factor, :math:`G_0`", "Optional (default 1)"
   2, "Systematic phase offset, :math:`\phi_0` [deg]", "Optional (default 0)"
   3, "Time-variable gain factor, :math:`G_{\rm std}`, (std. deviation)", "Optional (default 0)"
   4, "Time-variable phase error, :math:`\phi_{\rm std}`, (std. deviation) [deg]", "Optional (default 0)"

Gain :math:`(G_0, G_{\rm std})` and phase :math:`(\phi_0, \phi_{std})`
parameters define a complex multiplicative factor applied to each detector
element. This complex factor is combined with the geometric beamforming weights
(i.e. weights that define the Array Factor) to give a set of weights used to
evaluate the station beam at each source direction.

As a result, the beamforming weight, :math:`W`, for a given beam direction
:math:`(\theta_b, \phi_b)`, detector position :math:`(x,y,z)` and time
:math:`t` is given by:

.. math::

   W(\theta_b, \phi_b, x, y, z, t) =
   W_{\rm geometric} (\theta_b, \phi_b, x, y, z, t)
   (G_0 + G_{\rm error})  \exp\left\{ i (\phi_0 + \phi_{\rm error} )\right\}

where :math:`G_{\rm error}` and :math:`\phi_{\rm error}` are pseudo-random
values picked, at each time-step, from Gaussian distributions with standard
deviations :math:`G_{\rm std}`  and :math:`\phi_{\rm std}` respectively.


.. _telescope_element_cable_length_errors:

Element Cable Length Error Files
================================
In each station directory, there may be optionally a ``cable_length_error.txt``
file to specify the per-element cable length errors in metres.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description"
   :widths: 15, 85

   1, "Cable length error, in metres."

.. raw:: latex

    \clearpage


.. _telescope_element_apodisation:

Element Apodisation Files
=========================
In each station directory, there may be optionally an ``apodisation.txt``
(or ``apodization.txt``) file to specify additional complex multiplicative
beamforming weights to modify the shape of the station beam. If present,
these weights are multiplied with the DFT weights calculated for the beam
direction required at each time-step.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description", "Comment"
   :widths: 15, 42, 43

   1, "Element multiplicative weight (real part)", "Optional (default 1)"
   2, "Element multiplicative weight (imaginary part)", "Optional (default 0)"


.. _telescope_element_feed_angle:

Element Feed Angle Files
========================
In each station directory, there may be optionally ``feed_angle.txt``,
``feed_angle_x.txt`` and/or ``feed_angle_y.txt`` files
to specify the Euler angles of the feeds of the nominal X and Y dipoles.
If only a single ``feed_angle.txt`` file is present, the same data are used
for both the X and Y polarisations. Note that all the angles represent
differences from zero, which is the ideal case where both dipoles are
orthogonal and in the plane of the station.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description"
   :widths: 15, 85

   1, "Euler angle alpha around z-axis [deg]"


.. .. _telescope_mount_type:

   Mount Type (Not implemented)
   ============================
   In each station directory, there may be optionally a ``mount_type.txt``
   file to specify the mount type of the station platform.

   Allowed types are 'Fixed' (where the elements are fixed to the
   ground, as in an aperture array), 'Alt-az' (where the elements are mounted on
   a platform that tracks the sky relative to the horizon) or 'Equatorial' (where
   the elements are mounted on a platform that tracks the sky relative to the
   equator). Only the first letter of each type is checked.

   If the file is omitted, the station platform has a mount type of 'Fixed'.

   In order, the parameter columns are:

   .. csv-table::
      :header: "Column", "Description"
      :widths: 15, 85

      1, "Station platform mount type (character: either 'A' 'E' or 'F')."


.. _telescope_permitted_beams:

Permitted Beam Directions
=========================
In each station directory, there may be optionally a ``permitted_beams.txt``
file to specify a list of azimuth and elevation coordinates for all local
beam directions permitted at that station. If the file is omitted, it is
assumed that the station can form a beam anywhere on the sky. If the file is
present, then the nearest permitted direction to the computed phase centre
will be selected for each time step.

.. The permitted beam directions are defined with respect to the plane
   of the station platform. Use caution if the mount type of the station
   platform is not 'Fixed', as these angles will then not correspond to azimuth
   and elevation in the normal sense.

In order, the parameter columns are:

.. csv-table::
   :header: "Column", "Description"
   :widths: 15, 85

   1, "Azimuth coordinate of beam (local East from North) [deg]"
   2, "Elevation coordinate of beam (relative to local horizon) [deg]"


.. raw:: latex

    \clearpage

.. _telescope_gain_model:

Gain Model
==========
Externally-generated complex gains can be specified for the antennas in
each station, and/or for each station in the interferometer, as a function
of time, frequency and polarisation. If supplied, these complex gains must be
written to a HDF5 file called ``gain_model.h5`` and saved in the appropriate
station or telescope model directory. The HDF5 file must contain 3 datasets
under the root group, with the following names and dimensions:

* ``freq (Hz)`` is a 1-dimensional array containing a list of
  frequencies (in Hz) for each channel in the gain table.
  The length of this array must be the same as the channel dimension
  in the following two arrays.

* ``gain_xpol`` is a 3-dimensional array of complex gains for the
  X-polarisation, with the three dimensions representing
  (time, channel, antenna/station), where the time index is the slowest
  varying, and the antenna or station index is the fastest varying.
  Since HDF5 does not support complex types natively, each element of
  this array must be a compound type of two floating-point values,
  which represent the real and imaginary parts of the gain.
  The time index in this array corresponds to the time index of each
  snapshot in the simulation, so the gain table should be tailored to
  the observation parameters.
  The appropriate channel index will be selected using the list of frequencies,
  by finding the nearest frequency in the table to the frequency of each
  channel.
  If the gain values do not vary with time or channel, the size of the
  corresponding dimension should be set to 1.
  The size of the antenna (or station) dimension **must** match the
  number of antennas (or stations) specified in the layout file in the
  same directory.

* ``gain_ypol`` is the corresponding 3-dimensional array of complex gains
  for the Y-polarisation. It may be omitted if running simulations in scalar
  mode, or if the values should be the same for both polarisations.

.. _telescope_harp:

HARP Data
=========
When compiled with the ``harp_beam`` library, OSKAR can use coefficients
exported by the HARP electromagnetic simulation package to evaluate station
beams directly, without needing to first evaluate individual element patterns.
This method of beam evaluation is generally much faster than other methods
if mutual coupling needs to be taken into account.

Inside each station directory, one or more HDF5 files containing the
coefficients need to be supplied as a function of frequency. The filename
must contain the word "HARP", and the last number in the filename will be
interpreted as the frequency in MHz for which the coefficients apply.
(For example, "HARP_100.h5" and "data_HARP_SKALA4_rand256_100MHz.h5" are
equivalent.)

The following attributes and datasets must be present inside each HDF5 file:

* Attribute ``freq`` (floating-point): the simulated frequency in Hz.

* Attribute ``num_ant`` (integer): the number of antennas in the station.

* Attribute ``num_mbf`` (integer): the number of macro basis functions
  per antenna.

* Attribute ``max_order`` (integer): the maximum order of the spherical-wave
  decomposition (SHD) of the MBF patterns.

* Dataset ``alpha_te``: 2D complex matrix of size
  (``num_mbf``, ``max_order`` * (2 * ``max_order`` + 1)) containing the
  coefficients of the TE spherical modes of the MBF patterns.

* Dataset ``alpha_tm``: As above, but for the TM modes.

* Dataset ``coeffs_pola``: 2D complex matrix of size
  (``num_ant``, ``num_mbf`` * ``num_ant``) containing the MBF coefficients of
  each embedded element pattern, associated with the receiving port A (or X).

* Dataset ``coeffs_polb``: As above, but for port B (or Y).

If the station model directory has one of these HDF5 files, OSKAR will use
the HARP beam evaluation method instead of the default one.

.. raw:: latex

    \clearpage

.. _telescope_element_patterns:

Element Pattern Files
=====================

.. _telescope_element_patterns_numerical:

Numerical Element Patterns
--------------------------
Numerically-defined antenna element pattern data can be used for the
simulation. OSKAR currently supports the loading of ASCII text files produced
by the CST (Computer Simulation Technology) software package. Since
version 2.7.0, either the theta-phi or the Ludwig-3 polarisation system can
be used to represent the data. These files must contain eight columns,
in the following order:

#. Theta [deg]
#. Phi [deg]
#. Abs dir *
#. Abs theta (if theta-phi) / Abs horizontal (if Ludwig-3)
#. Phase theta [deg] (if theta-phi) / Phase horizontal [deg] (if Ludwig-3)
#. Abs phi (if theta-phi) / Abs vertical (if Ludwig-3)
#. Phase phi [deg] (if theta-phi) / Phase vertical [deg] (if Ludwig-3)
#. Ax. ratio *

(Columns marked * are ignored during the load, but must still be present.)

Ludwig-3-format data are detected by the presence of the word "Horiz" on the
first (header) line of the file; otherwise, the theta-phi system is assumed.

Since version 2.6.0, "unpolarised" (scalar) numerical element pattern data
files can be supplied, and these will be used if OSKAR is running in a scalar
or Stokes-I-only mode. Data files for scalar numerical element responses must
contain three or four columns, in the following order:

#. Theta [deg]
#. Phi [deg]
#. Amplitude
#. Phase [deg] (optional)

Before being used in the simulation, the element pattern data must be fitted
with B-splines. The fitting procedure is performed using the
``oskar_fit_element_data`` application which is built as part of the
OSKAR package. Please see the settings file documentation for a description
of the options used by this application.

To be recognised and loaded, the fitted element data must be supplied in
files that use the following name pattern, which is created
automatically by the fitting procedure:

.. code-block:: text

   element_pattern_fit_[x|y|scalar]_<element type index>_<frequency in MHz>.bin

**The element type index should be 0 unless there is more than one type of
element in the station** (as specified in the station's ``element_types.txt``),
and the frequency is the frequency in MHz for which
the element pattern data are valid: so, for example,
``element_pattern_fit_x_0_600.bin`` would contain fitted
coefficients to the data for the first type of X-dipole at 600 MHz.
The frequency nearest to the current observing frequency is used when
evaluating the response.

These files define the patterns used for the nominal X- and Y-dipoles.
The location of these files defines their scope: if placed in the top-level
directory, then they are used for all stations; if placed in a station
directory, they are used only for that station. In this way, it is possible
to specify different element patterns for each station.

.. note::

   Surfaces are fitted to the numerically-defined antenna data using bicubic
   B-splines. Since the quality of the fit depends critically on the fitting
   parameters (adjustable using the OSKAR settings file), **it is essential that
   each fitted surface is inspected graphically to ensure that there are no
   artefacts introduced by the fitting process**. This can be done by saving a
   FITS image of the element pattern (created by evaluating the fitted
   coefficients) by making an image of the station beam from a single-element
   station.

.. _telescope_element_patterns_spherical_wave:

Spherical Wave Element Patterns
-------------------------------
Since OSKAR 2.7.5, spherical wave coefficients can be used to represent
element patterns. Coefficients should be supplied in text files in the
telescope or station model folders, with the following naming convention:

.. code-block:: text

   element_pattern_spherical_wave_[x|y]_[te|tm]_[re|im]_
       <element type index>_<frequency in MHz>.txt

**The element type index should be 0 unless there is more than one type of
element in the station** (as specified in the station's ``element_types.txt``),
and the frequency is the frequency in MHz for which
the element pattern data are valid.
The frequency nearest to the current observing frequency is used when
evaluating the response.

Separate files are needed both for the real and imaginary parts and the
TE and TM modes. The X and Y labels are optional,
and can be used if the two polarisations are different.

For example, ``element_pattern_spherical_wave_x_te_re_0_100.txt`` would
contain the real part of the coefficients for the TE mode for the first
type of X-polarised antenna at 100 MHz.

Each line in each file contains all the data for one order of theta
(one value of :math:`l`, starting at :math:`l=1`);
the number of lines in the files gives the maximum order in theta to
use (:math:`l_{\rm max}`), and the number of terms on each
line is :math:`2 l_{\rm max} + 1`.
Only the first :math:`2 l + 1` terms on each line are used, but each line
also needs trailing zeros for values :math:`l < l_{\rm max}`.

.. Functional Element Patterns (Not implemented)
   -------------------------------------------------

   Since version x.x.x, functionally-defined element patterns can be specified
   for a given element type, so that functional and numerical elements can be
   mixed within a station.

   To be recognised and loaded, the functional element data must be supplied in
   files that use the following name pattern:

   ``element_pattern_[x|y]_<element type index>.txt``

   **The element type index should be 0 unless there is more than one type of
   element in the station** (as specified in the station's ``element_types.txt``):
   so, for example, ``element_pattern_x_1.txt`` would contain the functional
   pattern for the *second* type of X-dipole (and for this to actually be
   used in a simulation for this example, at least one element in the station
   would need to have a type index of 1 rather than 0, where type 0 represents
   the default element type).

   The content of the file must be a single line, which can contain
   columns in the following order:

   - **Element base type code.**
     This is the base type of the element pattern. Currently supported type
     codes are 'I' (for isotropic) or 'D' (for dipole).
   - **(For a dipole element) The length of the dipole.**
   - **(For a dipole element) The dipole length unit code.**
     Supported type codes are 'M' for metres or 'W' for wavelengths.
   - **Taper type code.**
     This is the type of taper applied on top of the base type. Currently
     supported type codes are 'N' (no taper; the default), 'C' (cosine)
     or 'G' (Gaussian).
   - **(For a cosine taper) The power of the cosine.**
   - **(For a Gaussian taper) The FWHM of the Gaussian, in degrees, at the
     reference frequency.**
   - **(For a Gaussian taper) The reference frequency, in Hz.**

   ** Note that if a column is not relevant for a particular option, it must be
   omitted.**

   These files define the patterns used for the nominal X- and Y-dipoles.
   The location of these files defines their scope: if placed in the top-level
   directory, then they are used for all stations; if placed in a station
   directory, they are used only for that station.

   If no element pattern is specified for a station, the default behaviour is
   to use an un-tapered half-wavelength dipole.

   Examples
   ^^^^^^^^

   An isotropic (unpolarised) element pattern with a Gaussian FWHM taper
   of 20 degrees at 100 MHz would specify ``I G 20.0 100e6``, while a
   half-wavelength dipole with a :math:`\cos^{2}` taper would specify
   ``D 0.5 W C 2.0``.


.. raw:: latex

    \clearpage

.. _telescope_noise:

System Noise Configuration Files
================================

OSKAR telescope models may contain files, which, if present, can be used
to specify the addition of uncorrelated system noise to interferometry
simulations.

For details of how uncorrelated noise is added to interferometry simulations,
please refer to :ref:`theory_noise` in the Theory of Operation document.
It should be noted that simulation settings files control the use and
selection of noise files within a telescope model. A description of these
settings can be found in the interferometry section of the OSKAR
:ref:`settings` documentation.

Noise files can be placed either in the top-level telescope model folder, or
separately in each station folder to allow for a different level of noise to be
added for each station. These files specify lists of values in plain text,
with successive values in the list separated by a new line. As with other
OSKAR plain text format configuration files, lines starting with a hash ``#``
character are treated as comments, and empty lines are ignored.

The name and contents of each file type are described below.

* ``noise_frequencies.txt``

  A list of frequencies, in Hz, for which noise values are defined. This file
  should be situated in the root of the telescope model directory structure.

.. code-block:: text

   # Example noise_frequencies.txt file
   #
   # This file contains a list of frequencies, in Hz, for which noise values
   # are defined.
   #

   50.0e6
   60.0e6
   70.0e6
   80.0e6

* ``rms.txt``

  A list of noise flux density RMS values, in Jy, as a function of frequency.
  The number of RMS values in the list should match the number of specified
  noise frequencies. Files can be situated in the root of the telescope model
  directory or in the top-level station folders. Files in station directories
  allow a different RMS values to be specified per station, and files in the
  root directory allow a quick way to specifying common RMS values for the
  entire array.

.. code-block:: text

   # Example rms.txt file
   #
   # This file contains a list of Gaussian RMS values, in Jy, from which
   # noise amplitude values are evaluated. Entries in the list correspond
   # to the noise RMS value at the frequency defined either by the
   # corresponding line in the noise_frequencies.txt file, or by the
   # frequency specification in the noise settings.
   #

   0.7
   0.5
   0.3
   0.2
