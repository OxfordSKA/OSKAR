1. Introduction
--------------------------------------------------------------------------------

This module provides an API for reading and writing OSKAR binary files. 
It can be built either as part of the OSKAR package, or as a stand-alone 
library.

The motivation for supporting a stand-alone interface to OSKAR binary files
is driven by the realisation that application developers may have the need for
a simple API for interfacing with OSKAR binary file format data products
from their own code.

As this API is envisaged to change relatively little over time, any developer 
wishing to interface with OSKAR binary data files is strongly advised to use 
this rather than attempting to use the unsupported higher level functions in 
the OSKAR package, which are subject to far more frequent API changes.


2. Stand-alone mode build instructions
--------------------------------------------------------------------------------

This module uses a standard CMake build system, and can be compiled and 
installed with the following commands:

    mkdir build
    cd build
    cmake ../path/to/oskar/binary/file/CMakeLists.txt [OPTIONS]
    make

The module can then be installed with:

    make install

A number of options can be passed to the cmake command. Of note:

    -DCMAKE_BUILD_TYPE=<release|debug> (default: release)
        Determines if the code should be built with release or debug flags.

    -DCMAKE_INSTALL_PREFIX=<path> (default: /usr/local)
        Sets the path prefix with which to install the library and
        associated header files.


3. Tests & examples
--------------------------------------------------------------------------------

Some tests and examples are provided in the test folder. An example showing
how to read an OSKAR visibility format binary file is shown in:

    Test_binary_vis_read_write.c.

By default, the tests are built along with the module and can be run either 
directly, or via the CMake test tool as follows:

    ctest [--verbose]

The tests should pass. If any fail, please report this by copying
the terminal output and sending it, together with your operating system 
version and your version of the OSKAR binary library, to the following 
email address:

    oskar@oerc.ox.ac.uk
