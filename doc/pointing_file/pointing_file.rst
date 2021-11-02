.. _pointing_file:

*************
Pointing File
*************

It may be necessary to simulate arrays where each station (and/or sub-station,
if using hierarchical beamforming schemes) has a different phase centre. The
pointing file described in this document provides the means to do this.

File Format
===========
The pointing file is a plain-text file that can be used to specify the
direction of the beamformed phase centre for every (sub-) station in the
array.

.. tip::

   This is an optional file: if it is not specified, all station beams will
   point in the direction of the interferometer phase centre, as defined in the
   main settings file.

Each row in the file is used to define a beam direction. The text file has a
variable number of columns per row, which specify the address of the
station(s) in the hierarchy (via multiple indices) and the beam direction to
set for the station(s). The columns are:

- The index of the top-level station.
- The index of the station (or tile) at the next level down, if required.
- ... (and so on, for further sub-stations, if required).
- The coordinate system used for the beam specification. This is a string
  that may be either AZEL or RADEC to specify horizontal or equatorial
  coordinates.
- The longitude of the beam in degrees.
- The latitude of the beam in degrees.

Wildcards (an asterisk, ``*``) may be used in the index columns to allow the
same direction for all stations of the specified parent station.

**An entry in the file will set the beam direction for the station(s) at the
last specified index, and recursively for all child stations.**

.. note::

   Note also that the order in which lines appear in the file is important.
   Entries that appear later override those that appear earlier.

Example
-------
For example, a file may contain the following lines to specify different phase
centres for beams formed at the tile and station levels:

.. code-block:: text

   *   RADEC 45.0 60.0 # All stations (and children) track (RA, Dec) = (45, 60).
   3   RADEC 45.1 59.9 # Station 3 (and children) tracks (RA, Dec) = (45.1, 59.9).
   * * AZEL  60.0 75.0 # All tiles in all stations have fixed beams.
   0 * AZEL  60.1 75.0 # All tiles in station 0 are offset from the rest.
   2 6 AZEL   0.0 90.0 # Tile 6 in station 2 is pointing at the zenith.
