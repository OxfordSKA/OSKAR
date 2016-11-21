
# 1. Introduction

The OSKAR simulation package can be built and installed by following the steps
described below. The current version is available as a source code distribution,
targeted at Linux and Mac OS X operating systems. Partial installation under
Microsoft Windows may be possible, but is not currently supported.


# 2. Dependencies

OSKAR depends on a number of other libraries. The main components of OSKAR
require CUDA 5.5 or later and Qt 4.6 or later to be installed on the target
system. Additionally, the casacore libraries must be present if
Measurement Sets are to be exported.

Please ensure that the required dependencies are installed before proceeding
further.

The list below summarises the main dependencies:

* CMake (http://www.cmake.org), version >= 3.0.0
* NVIDIA CUDA (http://developer.nvidia.com/cuda-downloads), version >= 5.5
* Qt4 (http://www.qt.io), version >= 4.6
* casacore (https://github.com/casacore/casacore), version >= 1.5.0


# 3. Building OSKAR

OSKAR can be built by issuing the following commands:

    $ mkdir build
    $ cd build
    $ cmake [OPTIONS] ../path/to/top/level/source/folder
    $ make

OSKAR can then be installed with:

    $ make install


## 3.1. Build Options

When running the 'cmake' command a number of build options can be specified.
These are listed below.

    * -DCUDA_ARCH=<arch> (default: all)
        Sets the target architecture for the compilation of CUDA device code.
        <arch> must be one of either: 1.3, 2.0, 2.1, 3.0, 3.5, 3.7,
                                      5.0, 5.2, 6.0, 6.1 or ALL.
        ALL is for all Fermi, Kepler, Maxwell and Pascal architectures (>= 2.0).

    * -DCMAKE_INSTALL_PREFIX=<path> (default: /usr/local/)
        Path prefix used to install OSKAR (with make install).

Advanced Build Options:

    * -DCMAKE_BUILD_TYPE=<release or debug> (default: release)
        Build OSKAR in release or debug mode.

    * -DCASACORE_LIB_DIR=<path> (default: searches the system library paths)
        Specifies a custom path in which to look for the CasaCore libraries
        (libcasa_ms.so and others).
        Note: This should only be used in special cases, where the version of
        CasaCore installed in the system library path can't be used to build
        OSKAR.

    * -DCASACORE_INC_DIR=<path> (default: searches the system include paths)
        Specifies a custom path in which to look for the CasaCore library
        headers. This is the path to the top level casacore include folder.
        Note: This should only be used in special cases, where the version of
        CasaCore headers installed in the system include path can't be used to
        build OSKAR.

    * -DNVCC_COMPILER_BINDIR=<path> (default: None)
        Specifies a nvcc compiler binary directory override. See nvcc help.
        Note: This is likely to be needed only on OS X when the version of the
        compiler picked up by nvcc (which is related to the version of XCode
        being used) is incompatible with the current version of CUDA.

    * -DFORCE_LIBSTDC++=ON|OFF (default: OFF)
        If ON forces the use of libstdc++ with the Clang compiler.
        Note: Used for controlling linking behaviour when using clang
        or clang-omp compilers with dependencies which may have been compiled
        against libstdc++

    * -DBUILD_INFO=ON|OFF (default: OFF)
        If ON enables the display of diagnostic build information when
        running CMake.

    * -DPYTHON_LIBRARY=<path to library> (default: None)
        Path to the Python library which can be used to specify a specific
        installation of Python to use.

    * -DPYTHON_INCLUDE_DIR=<path to Python.h> (default: None)
        Path to Python.h which can be used to specify a specific installation
        of Python to use.

## 3.2. Custom (Non-System) Qt4 Installations

When searching for a valid Qt4 installation, the OSKAR CMake build system
queries the qmake binary in order to determine the location of the relevant
libraries and headers. Therefore, all that is required to use a specific
version of Qt4 is to add the location of the desired qmake binary to the
beginning of the system search path.


# 4. Testing the Installation

## 4.1 Unit Tests

The unit test binaries can be run by typing the following command from the
build directory:

    $ ctest [--verbose]

All the unit tests should pass. If any fail, please report this by copying
the terminal output and sending it, together with a description of the
hardware in your machine, your operating system version and your version of
OSKAR, to the following email address:

    oskar@oerc.ox.ac.uk

## 4.2 Running the Example Simulation

With any fresh install of OSKAR, we recommend running the
example simulation described in the documentation found at

    http://oskar.oerc.ox.ac.uk/

to establish if a simple simulation behaves as expected.
