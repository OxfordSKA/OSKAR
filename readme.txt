--------------------------------------------------------------------------------
| OSKAR 2.0.0-beta                                  last update: 16 April 2012 |
--------------------------------------------------------------------------------



1. OSKAR: The Open Square Kilometre Array Radio Telescope Simulator
--------------------------------------------------------------------------------

The OSKAR package consists of a number of open source libraries and applications
for the simulation of radio interferometers. OSKAR has been designed with a
focus on simulating large aperture arrays, such as that of the SKA.


2. OSKAR repository structure
--------------------------------------------------------------------------------


2.1 Top level source folder structure
-------------------------------------

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
 |--- math             : Generic math functions.
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




2.2 Sub-folder structure
------------------------

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


2.3 Libraries built
-------------------

* oskar                 : Main oskar simulation library.

* dierckx               : Spline library.

* oskar_apps            : Application utility library.

* oskar_fits            : FITS interface library.

* oskar_ms              : Measurement set writer.

* oskar_widgets         : Qt Widgets.



3. Notes: The OSKAR MATLAB interface
--------------------------------------------------------------------------------

3.1 Tab completion for mex functions
-------------------------------------

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
