oskar-lib repository structure (last update: 8 Aug 2011)
================================================================================


Top level src folder structure:
===============================================================================

Top level folder structure for oskar-lib.

src
 |--- apps             : Application utility library and various applications
 |                       which make use of the rest of oskar.
 |
 |--- cmake            : Build system macros.
 |
 |--- fits             : FITS format interface library.
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
 |--- station          : Evaluation of beam pattern (E Jones),
 |                       station level processing, antenna level processing.
 |
 |--- utility          : Global headers, timer.
 |
 |--- widgets          : Qt widgets, plotting, GUI components.




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
                        : Dependencies: casacore.

* oskar_widgets         : Qt Widget set (plotting, general widgets)
                        : Dependencies: Qt4, Qwt5-Qt4

* oskar_apps            : Application utility library.
                        : Dependencies: oskar_widgets, oskar, Qt, Qwt5-Qt4

* oskar_fits            : FITS interface library.
                        : Dependencies: CFITSIO


Matlab interface
================================================================================
    scripts/matlab      : Matlab wrapper scripts.

Tab completion for mex functions:
---------------------------------

    Edit the TC.xml file in the MATLAB root toolbox/local directory.
    
    e.g. to add filename tab completion for the oskar load_source_file and
    oskar_visibilties_read functions add the following to the end of the file
    just before the <\TC> tag.
        
        <binding name="load_source_file">
            <arg argn="1" ctype="FILE"/>
        </binding>
  
        <binding name="oskar_visibilities_read">
            <arg argn="1" ctype="FILE"/>
        </binding>
        




Python interface
================================================================================
(Currently a placeholder - potentially useful as an interface with CASA)




