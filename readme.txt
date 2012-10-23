+------------------------------------------------------------------------------+
| OSKAR 2.1-beta                                  Last update: 23 October 2012 |
+------------------------------------------------------------------------------+

1. OSKAR: The Open Square Kilometre Array Radio Telescope Simulator
--------------------------------------------------------------------------------

The OSKAR package consists of a number of open source libraries and applications
for the simulation of astronomical radio interferometers. OSKAR has been
designed primarily to produce simulated data from large aperture arrays, such
as those envisaged for the SKA.


2. OSKAR Repository Structure
--------------------------------------------------------------------------------

2.1 Top-level Structure
--------------------------------------------------------------------------------

Top level folder structure for OSKAR.

 src
 |--- apps             : Applications directory containing application main()
 |    |                  functions.
 |    |
 |    |--- lib         : Application library. Utility functions used by OSKAR
 |                       applications.
 |
 |--- cmake            : Build system macros.
 |
 |--- extern           : External libraries used by OSKAR.
 |    |
 |    |--- dierckx     : Subroutines for calculating smoothing splines (tests).
 |
 |--- fits             : FITS format interface library.
 |
 |--- imaging          : Image/imaging functions.
 |
 |--- interferometry   : Interferometry functions, telescope model functions.
 |
 |--- math             : Generic math functions.
 |
 |--- matlab           : MATLAB interface to OSKAR.
 |
 |--- ms               : Measurement Set writer.
 |
 |--- sky              : Sky generators, coordinate conversion,
 |                       horizon clipping, sky model functions.
 |
 |--- station          : Evaluation of beam pattern (E Jones),
 |                       station level processing, antenna level processing.
 |
 |--- utility          : Utility functions.
 |
 |--- widgets          : GUI components.


2.2 Sub-folder Structure
--------------------------------------------------------------------------------

Each of the top level module folders listed above will contain the following
folders.

(module)
 |--- src       : C/C++ source code for module.
 |
 |--- test      : module tests / unit tests.
 |    |
 |    |---src


3. OSKAR Components
--------------------------------------------------------------------------------

3.1 Libraries
--------------------------------------------------------------------------------
* oskar                    : Main OSKAR simulation library.
* oskar_apps               : Application utility library.
* oskar_fits               : FITS interface library.
* oskar_ms                 : Measurement Set writer.
* oskar_widgets            : Qt Widgets.

3.2 Applications
--------------------------------------------------------------------------------
* oskar                    : Main OSKAR GUI Application.
* oskar_sim_interferometer : Command line application for interferometry simulations.
* oskar_sim_beam_pattern   : Command line application for beamforming simulations.
* oskar_imager             : Command line application for simple DFT imaging.

3.3 MATLAB Interface functions
--------------------------------------------------------------------------------
An experimental MATLAB interface consisting of functions for:
* Reading OSKAR visibility files
* Reading OSKAR image files.
* Making images by DFT.
* Reading OSKAR source catalog files.
* Reading OSKAR binary files.

3.4 Unit tests
--------------------------------------------------------------------------------
A number of unit test binaries are built, and can be found in the module/test
directories.
