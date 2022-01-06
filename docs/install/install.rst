.. _install_guide:

******************
Installation Guide
******************

OSKAR can be installed by following the steps described below.
A source code archive, and pre-built binary packages for Linux (using
Singularity), macOS and Windows platforms are available to download from
https://github.com/OxfordSKA/OSKAR/releases

Platforms
=========

Linux
-----

Singularity Image for Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pre-built Singularity (`<https://sylabs.io/singularity/>`_) SIF container
image is available for Linux which can be used to run OSKAR command line
applications or Python scripts directly, without needing to compile or install
anything. For Singularity 3.0 or later, an application or script can be run
using the downloaded container with the `singularity exec` command,
which takes the form:

  .. code-block:: text

     singularity exec [flags] <container_path> <app_name> <arguments>...

Use the ``--nv`` flag to enable NVIDIA GPU support in Singularity, if
applicable.

Note also that Singularity will mount the home directory into the container by
default, unless configured otherwise. If you have packages installed in your
home area that should be kept isolated from those in the container (for
example, because of conflicting packages or Python versions, or if you see
other errors caused by trying to load wrong versions of shared libraries when
starting the container) then it may be necessary to disable this either by
using the ``--no-home`` flag, or re-bind the home directory in the container
to somewhere other than your actual $HOME using the ``-H`` flag.

As an example, to run the application ``oskar_sim_interferometer``
with a parameter file ``settings.ini`` and a container image file
``OSKAR-Python3.sif`` (both in the current directory) on a GPU use:

  .. code-block:: text

     singularity exec --nv ./OSKAR-Python3.sif \
          oskar_sim_interferometer settings.ini

Similarly, to run a Python script ``sim_script.py`` that uses OSKAR:

  .. code-block:: text

     singularity exec --nv ./OSKAR-Python3.sif python3 sim_script.py


Installation on Linux
^^^^^^^^^^^^^^^^^^^^^

To install the OSKAR package on a Linux system that does not have Docker or
Singularity, you will need to compile it from source. Ensure the dependencies
have been installed as described in `Dependencies`_ (below).
Then download the archive and follow the short steps in `Build Commands`_.

If using a GPU on Linux, please ensure you have an up-to-date driver for it.
Drivers can often be installed from your distribution's repository,
but may not always support the latest hardware.
NVIDIA drivers can be downloaded from `<https://www.geforce.com/drivers>`_

Uninstallation on Linux
^^^^^^^^^^^^^^^^^^^^^^^

To remove the OSKAR package on Linux, delete the applications, libraries and
headers installed by the ``make install`` step.
By default, these will be placed in:

* Applications: /usr/local/bin/oskar*
* Libraries: /usr/local/lib/liboskar*
* Headers: /usr/local/include/oskar/


macOS
-----

Installation on macOS
^^^^^^^^^^^^^^^^^^^^^

To install the OSKAR package on macOS, download and open the disk image (DMG)
file and drag the OSKAR.app bundle to your /Applications folder.
After installation, double-click the OSKAR.app bundle to launch the GUI and
set symbolic links to the applications in /usr/local/bin.

If using a GPU on macOS, please ensure you have an up-to-date driver for it.
NVIDIA drivers for macOS can be downloaded from
`<http://www.nvidia.com/object/mac-driver-archive.html>`_

Uninstallation on macOS
^^^^^^^^^^^^^^^^^^^^^^^

To remove the OSKAR package on macOS, delete the OSKAR.app bundle from
your /Applications folder, and delete symbolic links to the applications
by typing ``rm -f /usr/local/bin/oskar*`` from a terminal prompt.


Windows
-------

Installation on Windows
^^^^^^^^^^^^^^^^^^^^^^^

To install the OSKAR package on 64-bit Windows, download and run the
installer executable and follow the on-screen instructions.
After installation, the GUI can be launched using the shortcut on the
Desktop or Start Menu.

If using a GPU on Windows, please ensure you have an up-to-date driver for it.
NVIDIA drivers can be downloaded from `<https://www.geforce.com/drivers>`_

Uninstallation on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^

To remove the OSKAR package on Windows, uninstall it from the list of
software in Control Panel in the usual way.

.. raw:: latex

    \clearpage


Building OSKAR
==============

This section describes the steps needed to build and install OSKAR from source.

Dependencies
------------

If hardware acceleration is required, be sure to install appropriate GPU
drivers which are supported by the hardware manufacturer. Third-party graphics
drivers are unlikely to work.

When building from source, the only required dependency is
`CMake <https://cmake.org>`_.
All other dependencies are optional, but functionality will be
limited if these are not found by CMake.
**Note that these dependencies are required only if building from source**,
not if using a pre-built package.

The dependencies are:

* CMake >= 3.1 (`<https://cmake.org>`_)

* [Optional] CUDA >= 7.0 (`<https://developer.nvidia.com/cuda-downloads>`_)
  or OpenCL, required for GPU acceleration on supported hardware.

* [Optional] Qt 5 (`<https://qt.io>`_),
  required to build the graphical user interface.

* [Optional] casacore >= 2.0 (`<https://github.com/casacore/casacore>`_),
  required to use CASA Measurement Sets.

* [Optional] HDF5 >= 1.10 (`<https://www.hdfgroup.org>`_),
  required to use HDF5 files.

Packages for these dependencies are available in the package repositories
of many recent Linux distributions, including Debian and Ubuntu.

Build Commands
--------------

To build from source, either clone the repository (for the current master
branch) or download and unpack the source archive, then:

  .. code-block:: bash

     mkdir build
     cd build
     cmake [OPTIONS] ../top/level/source/folder
     make -j8
     make install

Build Options
^^^^^^^^^^^^^

When running the 'cmake' command a number of build options can be specified.

- ``-DCUDA_ARCH="<arch>"`` (default: all)

  - Sets the target architecture for the compilation of CUDA device code
  - <arch> must be one of either: 2.0, 2.1, 3.0, 3.2, 3.5, 3.7,
    5.0, 5.2, 6.0, 6.1, 6.2, 7.0, 7.5, 8.0, 8.6, 8.7 or ALL
  - Note that ALL is currently most from 3.5 to 7.5.
  - Separate multiple architectures using semi-colons, if required
    (e.g. -DCUDA_ARCH="ALL;8.0").

- ``-DCMAKE_INSTALL_PREFIX=<path>``  (default: /usr/local/)

  - Path prefix used to install OSKAR (with make install)

Advanced Build Options
^^^^^^^^^^^^^^^^^^^^^^

- ``-DCASACORE_LIB_DIR=<path>`` (default: searches the system library paths)

  - Specifies a custom path in which to look for the casacore libraries
    (libcasa_tables.so and others) if they are not in the system library path.

- ``-DCASACORE_INC_DIR=<path>`` (default: searches the system include paths)

  - Specifies a location to search for the casacore library headers if they
    are not in the system include path.
    This is the path to the top level casacore include folder.

- ``-DCMAKE_PREFIX_PATH=<path>`` (default: None)

  - Specifies a location in which to search for Qt 5. For example, if
    using Homebrew on macOS, this may need to be set to /usr/local/opt/qt5/

- ``-DFIND_CUDA=ON|OFF`` (default: ON)

  - Can be used to tell the build system not to find or link against CUDA.

- ``-DFIND_OPENCL=ON|OFF`` (default: OFF)

  - Can be used to tell the build system not to find or link against OpenCL.
  - OpenCL support in OSKAR is currently experimental.

- ``-DNVCC_COMPILER_BINDIR=<path>`` (default: None)

  - Specifies a nvcc compiler binary directory override. See nvcc help.
  - Note: This is likely to be needed only on macOS when the version of the compiler picked up by nvcc (which is related to the version of XCode being used) is incompatible with the current version of CUDA.
  - Set this to 'clang' on macOS if using GCC to build the rest of OSKAR.

- ``-DFORCE_LIBSTDC++=ON|OFF`` (default: OFF)

  - If ON forces the use of libstdc++ with the Clang compiler.
  - Used for controlling linking behaviour when using clang or clang-omp compilers with dependencies which may have been compiled against libstdc++

- ``-DCMAKE_BUILD_TYPE=<release, debug>``  (default: release)

  - Build in release or debug mode.

- ``-DBUILD_INFO=ON|OFF`` (default: OFF)

  - If ON enables the display of diagnostic build information when running CMake.


Unit Tests
----------
After building from source, the unit tests should be run to make sure there
are no problems with the build.
(Note that pre-built packages do not include the unit tests.)

From the build directory, the unit tests can be run by typing:

  .. code-block:: bash

     ctest

Example Simulation
------------------
The :ref:`example simulation <example>` can be run to check that a
simple simulation behaves as expected.


Python Interface
================

After installing OSKAR, the :ref:`Python interface <python_interface>` to it
can be installed to make it easier to use from Python scripts.
Straightforward instructions for installation with ``pip`` can be found in the
``python`` subdirectory of the source distribution,
or on the :ref:`Python interface quick-start <python_quickstart>` page.
