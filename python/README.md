# The OSKAR Python Interface

The OSKAR package has been designed to produce simulated visibility data from
radio telescopes containing aperture arrays. The software is written
mainly in C and offers GPU acceleration using CUDA or OpenCL. The Python
bindings to OSKAR make it easy to run simulations using Python scripts.

Documentation for the OSKAR Python bindings can be found by following the link
on the [simulation tools page](https://developer.skatelescope.org/projects/sim-tools/en/latest/)
of the SKA Developer Portal.


## Installation

### Linux and macOS

 - Make sure you have a working Python environment
   (including `pip` and `numpy`), C and C++ compilers.
     - On macOS, the compilers are part of XCode, which is in the App Store.


 - Make sure OSKAR has been installed.
     - On macOS, you can drag the [pre-built package](https://github.com/OxfordSKA/OSKAR/releases) `OSKAR.app` to `/Applications`


 - Open a Terminal.


 - (Not usually required) If OSKAR is installed in a non-standard location,
   edit the paths in `setup.cfg` or temporarily set the two environment
   variables:

```bash
export OSKAR_INC_DIR=/path/to/oskar/include/folder
export OSKAR_LIB_DIR=/path/to/oskar/lib
```

 - Install the Python interface with:

```bash
pip install --user 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
```

 - The `--user` flag is optional, but you may need root permission without it.

### Windows

 - Make sure you have a working [Python](https://www.python.org/downloads/windows/)
   environment (including `pip` and `numpy`),
   and [Visual Studio Community C and C++ compiler](https://visualstudio.microsoft.com/vs/community/).
     - You will need to make sure that Python is added to the PATH environment
       variable when it is installed.
     - These steps also work with the Anaconda Python distribution,
       but Anaconda is not required.


 - Make sure OSKAR has been installed using the [pre-built package](https://github.com/OxfordSKA/OSKAR/releases).
     - In the installer, you will need to select the option **Add OSKAR to the PATH**,
       and install all optional components (headers and libraries).


 - Open a Command Prompt (or an Anaconda Prompt, if using Anaconda).


 - (Not usually required) If OSKAR is installed in a non-standard location,
   edit the paths in `setup.cfg` or temporarily set the two environment
   variables:

```
set OSKAR_INC_DIR=C:\path\to\oskar\include\folder
set OSKAR_LIB_DIR=C:\path\to\oskar\lib
```

 - Install the Python interface with:

```
pip install "git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python"
```

### Using Pipenv

This works also with [Pipenv](https://docs.pipenv.org)
(but make sure the above environment variables are set first, if necessary):

```bash
pipenv install -e 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
```


## Uninstallation

After installation using the steps above, the OSKAR Python interface can
be uninstalled using:

```bash
pip uninstall oskarpy
```

(This does not uninstall OSKAR itself, only the Python interface to it.)

