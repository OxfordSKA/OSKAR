Introduction
------------

OSKAR binary is an module providing a fairly low level API for for reading and
writing OSKAR binary files. It can be built either as part of the OSKAR
package or as a stand-alone library.

The motivation for supporting a stand-alone mode for the OSKAR binary file API
is driven by the realisation that application developers may have the need for
a simple API for interfacing with OSKAR binary file format data products
from their own code.

As this API is envisaged to change very little over time, any developer wishing
to interface with OSKAR binary data files is strongly advised to use this rather
than attempting to use the unsupported higher level functions in the OSKAR
package, which will be subject to far more frequency API changes.

Stand-alone mode build instructions
-----------------------------------

This module uses a relatively standard CMake build system and therefore can
be compiled and installed with the following commands.

    mkdir build
    cd build
    cmake ../path/to/oskar/binary/file/CMakeLists.txt [OPTIONS]
    make

The module can then be install with:

    make install

A number of options can be passed to the cmake command. Of note,

    -DCMAKE_BUILD_TYPE=<release|debug> (default = release)
        Determines if the code should be built with release or debug flags.

    -DCMAKE_INSTALL_PREFIX=<path> (default=/usr/local)
        Sets the path prefix with which to install the library and
        associated header files.


Tests & examples
----------------

A number of tests and examples (currently 2) are provided in the test folder.

By default, these are built along with the module and can be run using the
either directly or via the CMake test tool as follows:

    ctest [--verbose]

The tests should pass. If any fail, please report this by copying
the terminal output and sending it, together with a description of the
hardware in your machine, your operating system version and your version of
the OSKAR binary library, to the following email address:

    oskar@oerc.ox.ac.uk
