oskar-lib repository structure (last update: 8 Aug 2011)
================================================================================


Top level src folder structure:
===============================================================================

Top level folder structure for oskar-lib.

src
 |--- station          : Evaluation of beam pattern (E Jones),
 |                       station level processing, antenna level processing.
 |
 |--- cmake            : Build system macros.
 |
 |--- imaging          : FFT imager, simple post processing (CLEAN).
 |
 |--- interferometry   : Interferometry simulation wrappers.
 |
 |--- math             : Generic math functions.
 |
 |--- ms               : Measurement set writer.
 |
 |--- sky              : Sky generators, global to local coordinate conversion,
 |                       horizon clipping.
 |
 |--- utility          : Global headers, timer.
 |
 |--- widgets          : Qt widgets, plotting, GUI components.
 |
 |--- (uvfits)         : reserved placeholder for UVFITS output module.



Sub-folder structure
================================================================================

Each of the top level module folders listed above will contain the following
folders.

- module |--- cudak     : CUDA kernels.
         |    |
         |    |---src
         |
         |--- (matlab)  : reserved placeholder for MATLAB MEX functions.
         |
         |--- (python)  : reserved placeholder for Python modules.
         |
         |--- src       : C/C++ source code for module.
         |
         |--- test      : module tests / unit tests.
         |    |
         |    |---src


Libraries built
================================================================================

* oskar                 : Main oskar simulation library.
                        : Dependencies: CUDA(4.0+), CUBLAS, Thrust

* oskar_imaging         : Imaging library (& post processing)
                        : Dependencies: CUDA(4.0+), CUFFT, FFTW

* oskar_ms              : Measurement set writer
                        : Dependencies: CASA (ms)

* oskar_widgets         : Qt Widget set (plotting, general widgets)
                        : Dependencies: Qt4, Qwt5-Qt4

* (oskar_uvfits)        : (placeholder) UVFITS writer.
                        : Dependencies: CFITSIO


Matlab interface
================================================================================
    scripts/matlab      : Matlab wrapper scripts.


Python interface
================================================================================
(Currently a placeholder - potentially useful as an interface with CASA)




