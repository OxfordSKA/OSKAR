[![GitHub release](https://img.shields.io/github/release/OxfordSKA/OSKAR.svg?style=flat-square)](https://github.com/OxfordSKA/OSKAR/releases)

# OSKAR: A GPU-accelerated simulator for the Square Kilometre Array

OSKAR has been designed to produce simulated visibility data from radio
telescopes containing aperture arrays, such as those envisaged for the
Square Kilometre Array.

A source code archive, and pre-built binary packages for Linux (using
Singularity), macOS and Windows platforms are available to download from

- [https://github.com/OxfordSKA/OSKAR/releases](https://github.com/OxfordSKA/OSKAR/releases)

OSKAR is licensed under the terms of the 3-clause BSD License.
Please see the [LICENSE](LICENSE) file for details.

### Singularity image

A pre-built [Singularity](https://sylabs.io/singularity/) SIF container image
is available for Linux which can be used to run OSKAR command line
applications or Python scripts directly, without needing to compile or install
anything. For Singularity 3.0 or later, an application or script can be run
using the downloaded [container](https://github.com/OxfordSKA/OSKAR/releases)
with the `singularity exec` command, which takes the form:

    $ singularity exec [flags] <container_path> <app_name> <arguments>...

Use the `--nv` flag to enable NVIDIA GPU support in Singularity, if
applicable. As an example, to run the application `oskar_sim_interferometer`
with a parameter file `settings.ini` and a container image file
`OSKAR-Python3.sif` (both in the current directory) on a GPU use:

    $ singularity exec --nv ./OSKAR-Python3.sif oskar_sim_interferometer settings.ini

Similarly, to run a Python script `sim_script.py` that uses OSKAR:

    $ singularity exec --nv ./OSKAR-Python3.sif python3 sim_script.py

### Dependencies

If hardware acceleration is required, be sure to install appropriate GPU
drivers which are supported by the hardware manufacturer. Third-party graphics
drivers are unlikely to work.

When building from source, the only required dependency is
[CMake >= 3.1](https://cmake.org).
All other dependencies are optional, but functionality will be
limited if these are not found by CMake.
*Note that these dependencies are required only if building from source*, not
if using a [pre-built package](https://github.com/OxfordSKA/OSKAR/releases).

- [CMake >= 3.1](https://cmake.org)
- (Optional) [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
  or OpenCL, required for GPU acceleration on supported hardware.
- (Optional) [Qt 5](https://qt.io),
  required to build the graphical user interface.
- (Optional) [casacore >= 2.0](https://github.com/casacore/casacore),
  required to use CASA Measurement Sets.

Packages for these dependencies are available in the package repositories
of many recent Linux distributions, including Debian and Ubuntu.

### Build commands

To build from source, either clone the repository using
`git clone https://github.com/OxfordSKA/OSKAR.git` (for the current master
branch) or download and unpack the source archive, then:

    $ mkdir build
    $ cd build
    $ cmake [OPTIONS] ../path/to/top/level/source/folder
    $ make -j4
    $ make install

When running the 'cmake' command a number of options can be specified:

    * -DCUDA_ARCH="<arch>" (default: all)
        Sets the target architecture for the compilation of CUDA device code.
        <arch> must be one of either: 1.3, 2.0, 2.1, 3.0, 3.2, 3.5, 3.7,
                                      5.0, 5.2, 6.0, 6.1, 6.2, 7.0, 7.5
                                      or ALL.
        ALL is for all currently supported architectures (>= 3.0).
        Separate multiple architectures using semi-colons, if required.

    * -DCMAKE_INSTALL_PREFIX=<path> (default: /usr/local/)
        Path prefix used to install OSKAR (with make install).

#### Advanced build options

    * -DCASACORE_LIB_DIR=<path> (default: searches the system library paths)
        Specifies a location to search for the casacore libraries
        (libcasa_ms.so and others) if they are not in the system library path.

    * -DCASACORE_INC_DIR=<path> (default: searches the system include paths)
        Specifies a location to search for the casacore library headers if they
        are not in the system include path.
        This is the path to the top level casacore include folder.

    * -DCMAKE_PREFIX_PATH=<path> (default: None)
        Specifies a location to search for Qt 5 if it is not in a standard
        system path. For example, if using Homebrew on macOS, this may need
        to be set to /usr/local/opt/qt5/

    * -DFIND_CUDA=ON|OFF (default: ON)
        Can be used not to find or link against CUDA.

    * -DFIND_OPENCL=ON|OFF (default: OFF)
        Can be used not to find or link against OpenCL.
        OpenCL support in OSKAR is currently experimental.

    * -DNVCC_COMPILER_BINDIR=<path> (default: None)
        Specifies a nvcc compiler binary directory override. See nvcc help.
        This is likely to be needed only on macOS when the version of the
        compiler picked up by nvcc (which is related to the version of XCode
        being used) is incompatible with the current version of CUDA.
        Set this to 'clang' on macOS if using GCC to build the rest of OSKAR.

    * -DFORCE_LIBSTDC++=ON|OFF (default: OFF)
        If ON forces the use of libstdc++ with the Clang compiler.
        Used for controlling linking behaviour when using clang
        or clang-omp compilers with dependencies which may have been compiled
        against libstdc++

    * -DCMAKE_BUILD_TYPE=<release or debug> (default: release)
        Build in release or debug mode.

    * -DBUILD_INFO=ON|OFF (default: OFF)
        If ON enables the display of diagnostic build information when
        running CMake.

### Unit tests

From the build directory, the unit tests can be run by typing:

    $ ctest [--verbose]

### Python interface

After installing OSKAR, the Python interface to it can be installed to
make it easier to use from Python scripts.
Straightforward instructions for installation with `pip` can be
[found in the python subdirectory](python/README.md).

### Example simulation

The example simulation described in the
[documentation](https://github.com/OxfordSKA/OSKAR/releases)
can be run to check that a simple simulation behaves as expected.
