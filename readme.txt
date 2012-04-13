OSKAR repository structure (last update: 13 Apr 2012)
================================================================================


Top level source folder structure:
===============================================================================

Top level folder structure for OSKAR.

src
 |--- apps             : Applications directory containing application main()
 |    |                  functions. 
 |    |
 |    |--- lib         : Application library. Utility functions used by OSKAR
 |                       applications   
 |
 |--- cmake            : Build system macros.
 |
 |--- extern           : External libraries used by OSKAR.
 |    |
 |    |--- dierckx     : Subroutines for calculating smoothing splines 
 |
 |--- fits             : FITS format interface library.
 |
 |--- imaging          : Image/imaging functions.
 |
 |--- interferometry   : Interferometry functions.
 |
 |--- math             : Generic math funcitons.
 |
 |--- matlab           : Matlab interface to OSKAR.
 |
 |--- ms               : Measurement set writer.
 |
 |--- sky              : Sky generators, global to local coordinate conversion,
 |                       horizon clipping.
 |
 |--- station          : Evaluation of beam pattern (E Jones),
 |                       station level processing, antenna level processing.
 |
 |--- utility          : Utility functions.
 |
 |--- widgets          : GUI components.




Sub-folder structure
================================================================================

Each of the top level module folders listed above will contain the following
folders.

- module |--- cudak     : CUDA kernels.
         |    |
         |    |---src
         |
         |--- src       : C/C++ source code for module.
         |
         |--- test      : module tests / unit tests.
         |    |
         |    |---src


Libraries built
================================================================================

* oskar                 : Main oskar simulation library.

* dierckx               : Spline library.

* oskar_apps            : Application utility library.

* oskar_fits            : FITS interface library.

* oskar_ms              : Measurement set writer.

* oskar_widgets         : Qt Widgets.



MATLAB interface
================================================================================

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
        
