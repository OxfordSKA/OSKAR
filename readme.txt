+------------------------------------------------------------------------------+
| OSKAR 2.5                                          Last update: 14 July 2014 |
+------------------------------------------------------------------------------+

1. OSKAR: Oxford's Square Kilometre Array Radio-telescope simulator
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
 |-- apps             : Applications directory containing application main()
 |   |                  functions.
 |   |-- gui          : GUI classes.
 |   |-- lib          : Application library. Utility functions used by OSKAR
 |                      applications.
 |-- cmake            : Build system macros.
 |-- correlate        : Cross-correlation functions.
 |-- element          : Element structure and related functions.
 |-- extern           : External libraries used by OSKAR.
 |   |-- ezOptionParser-0.2.0: Library for command line option parsing.
 |   |-- gtest-1.7.0  : Google testing framework library.
 |-- fits             : FITS format interface library.
 |-- imaging          : Image/imaging functions.
 |-- interferometry   : Visibility functions and telescope model functions.
 |-- jones            : Jones data structure and jones matrix evaluation
 |                      functions.
 |-- math             : General math functions and DFT kernels.
 |-- matlab           : Experimental MATLAB interface.
 |-- measures         : Functions for coordinate conversions.
 |-- ms               : Measurement Set writer.
 |-- python           : Experimental Python interface.
 |-- settings         : Functions, widgets and structures for settings.
 |   |-- load         : Functions for loading settings ini files.
 |   |-- struct       : Settings structures and functions for handling them.
 |   |-- widgets      : Settings widgets for use with the OSKAR GUI
 |-- sky              : Horizon clipping and sky model functions.
 |-- splines          : Spline fitting functions.
 |-- station          : Evaluation of station beam pattern,
 |                      station level processing, antenna level processing.
 |-- utility          : Utility functions.
 |   |-- log          : Functions for printing formatted log messages.


2.2 Sub-folder Structure
--------------------------------------------------------------------------------

Each of the top level module folders listed above will contain the following
folders.

(module)
 |-- src       : C/C++ source code for module.
 |-- test      : module tests / unit tests.

3. OSKAR Components
--------------------------------------------------------------------------------

3.1 Libraries
--------------------------------------------------------------------------------
* oskar                    : Main OSKAR simulation library.
* oskar_apps               : Application utility library.
* oskar_fits               : FITS interface library.
* oskar_ms                 : Measurement Set writer.

3.2 Applications
--------------------------------------------------------------------------------
Simulation Applications:

* oskar                    : Main OSKAR GUI Application.
* oskar_sim_interferometer : Command line application for interferometry simulations.
* oskar_sim_beam_pattern   : Command line application for beamforming simulations.
* oskar_imager             : Command line application for simple DFT imaging.

Utility Applications:

* oskar_binary_file_query       :
* oskar_cuda_system_info        :
* oskar_fit_element_data        :
* oskar_fits_image_to_sky_model :
* oskar_image_stats             :
* oskar_image_summary           :
* oskar_settings_set            :
* oskar_settings_get            :
* oskar_vis_add                 :
* oskar_vis_add_noise           :
* oskar_vis_stats               :
* oskar_vis_summary             :
* oskar_vis_to_ascii_table      :
* oskar_vis_to_ms               :


3.3 MATLAB Interface functions
--------------------------------------------------------------------------------
An experimental MATLAB interface consisting of functions for:
* Reading OSKAR visibility files
* Reading OSKAR image files.
* Making images by DFT.
* Reading OSKAR source catalog files.
* Reading OSKAR binary files.

3.4 Python Interface
--------------------------------------------------------------------------------
Under development ...


3.5 Unit tests
--------------------------------------------------------------------------------
A number of unit test binaries are built, and can be found in the module/test
directories.
