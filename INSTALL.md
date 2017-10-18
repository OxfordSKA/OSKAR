
# 1. Introduction

OSKAR can be installed by following the steps described below.
A source code archive for Linux, and binary installer packages for
macOS and Windows platforms are available to download from:

    http://oskar.oerc.ox.ac.uk/


# 2. Dependencies

If GPU acceleration is required, an NVIDIA GPU with CUDA 5.5 or later
(and associated NVIDIA CUDA driver) must be installed.
If the graphical user interface (GUI) is required, Qt must also be installed.
Additionally, the casacore libraries must be present if Measurement Sets
are to be exported.

The dependencies are:

* CMake (https://cmake.org), version >= 3.1
* [Optional] NVIDIA CUDA (https://developer.nvidia.com/cuda-downloads), version >= 5.5
* [Optional] Qt 5 (https://www.qt.io)
* [Optional] casacore (https://github.com/casacore/casacore), version >= 1.5.0


# 3. Building OSKAR

To build OSKAR, open a terminal and type the following:

    $ mkdir build
    $ cd build
    $ cmake [OPTIONS] ../path/to/top/level/source/folder
    $ make

OSKAR can then be installed with:

    $ make install


## 3.1. Build Options

When running the 'cmake' command a number of build options can be specified.
These are listed below.

    * -DCUDA_ARCH="<arch>" (default: all)
        Sets the target architecture for the compilation of CUDA device code.
        <arch> must be one of either: 1.3, 2.0, 2.1, 3.0, 3.2, 3.5, 3.7,
                                      5.0, 5.2, 6.0, 6.1, 6.2, 7.0 or ALL.
        ALL is for all Kepler, Maxwell and Pascal architectures (>= 3.0).
        Multiple architectures can be specified by separating them with
        semi-colons.

    * -DCMAKE_INSTALL_PREFIX=<path> (default: /usr/local/)
        Path prefix used to install OSKAR (with make install).

Advanced Build Options:

    * -DCASACORE_LIB_DIR=<path> (default: searches the system library paths)
        Specifies a custom path in which to look for the CasaCore libraries
        (libcasa_ms.so and others).
        Note: This should only be used if the CasaCore library in the system
        library path can't be used to build OSKAR..

    * -DCASACORE_INC_DIR=<path> (default: searches the system include paths)
        Specifies a custom path in which to look for the CasaCore library
        headers. This is the path to the top level casacore include folder.
        Note: This should only be used if the CasaCore headers in the system
        include path can't be used to build OSKAR.

    * -DCMAKE_PREFIX_PATH=<path> (default: None)
        Specifies a location in which to search for Qt 5. For example, if
        using Homebrew on macOS, this may need to be set to /usr/local/opt/qt5/

    * -DCMAKE_BUILD_TYPE=<release or debug> (default: release)
        Build OSKAR in release or debug mode.

    * -DFIND_CUDA=ON|OFF (default: ON)
        Can be used to tell the build system not to find or link against CUDA.

    * -DNVCC_COMPILER_BINDIR=<path> (default: None)
        Specifies a nvcc compiler binary directory override. See nvcc help.
        Note: This is likely to be needed only on macOS when the version of the
        compiler picked up by nvcc (which is related to the version of XCode
        being used) is incompatible with the current version of CUDA.
        Set this to 'clang' on macOS if using GCC to build the rest of OSKAR.

    * -DFORCE_LIBSTDC++=ON|OFF (default: OFF)
        If ON forces the use of libstdc++ with the Clang compiler.
        Note: Used for controlling linking behaviour when using clang
        or clang-omp compilers with dependencies which may have been compiled
        against libstdc++

    * -DBUILD_INFO=ON|OFF (default: OFF)
        If ON enables the display of diagnostic build information when
        running CMake.

## 3.2. Custom (Non-System) Qt Installations

If Qt 5 cannot be found from the default system paths, make sure to set
CMAKE_PREFIX_PATH as described above.


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
