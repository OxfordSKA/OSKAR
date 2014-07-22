+------------------------------------------------------------------------------+
| OSKAR 2.5                                          Last update: 22 July 2014 |
+------------------------------------------------------------------------------+

1. OSKAR: Oxford's Square Kilometre Array Radio-telescope simulator
--------------------------------------------------------------------------------

The OSKAR package consists of a number of open source libraries and
applications for the simulation of astronomical radio interferometers.
OSKAR has been designed primarily to produce simulated data from large
aperture arrays, such as those envisaged for the SKA.


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
 |-- convert          : Functions for coordinate conversions.
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
 |-- src       : C or C++ source code for module.
 |-- test      : module tests / unit tests.


3. OSKAR Components
--------------------------------------------------------------------------------

3.1 Libraries
--------------------------------------------------------------------------------
* oskar                         : Main OSKAR library.
* oskar_apps                    : Application utility library.
* oskar_fits                    : FITS interface library.
* oskar_ms                      : Measurement Set interface library.

3.2 Applications
--------------------------------------------------------------------------------
Simulation Applications:

* oskar                         : Main OSKAR GUI application.
* oskar_sim_interferometer      : Command line application for interferometry
                                  simulations.
* oskar_sim_beam_pattern        : Command line application for beamforming
                                  simulations.
* oskar_imager                  : Command line application for simple DFT
                                  imaging.
* oskar_fit_element_data        : Command line application to fit splines to
                                  numerical element pattern data.

Utility Applications:

* oskar_binary_file_query       : Displays summary of contents of an OSKAR
                                  binary file.
* oskar_cuda_system_info        : Displays information about the CUDA devices
                                  found on the system.
* oskar_fits_image_to_sky_model : Creates an OSKAR sky model from a FITS image.
* oskar_image_stats             : Displays statistics for an OSKAR image.
* oskar_image_summary           : Displays a summary of an OSKAR image file.
* oskar_settings_set            : Sets the value for the given key in an
                                  OSKAR settings file.
* oskar_settings_get            : Returns the value of the given key in an
                                  OSKAR settings file.
* oskar_vis_add                 : Adds two OSKAR visibility files together.
* oskar_vis_add_noise           : Adds thermal noise to an OSKAR visibility
                                  file.
* oskar_vis_stats               : Displays statistics for an OSKAR visibility
                                  file.
* oskar_vis_summary             : Displays a summary of an OSKAR visibility
                                  file.
* oskar_vis_to_ms               : Converts an OSKAR visibility file to a
                                  Measurement Set.


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
