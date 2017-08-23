# The OSKAR Python Interface

## Installation

 1. Make sure OSKAR itself has been built and installed.
 2. If necessary, edit the file `setup.cfg` to set the paths to the
    OSKAR header and library installed on the system.
 3. From a terminal prompt in this directory, install the Python
    interface using `pip install --user .`
    - **Note the final dot on the line above!**
    - The `--user` flag is optional, but you may need root permission
      without it.

## Uninstallation

After installation using the steps above, the OSKAR Python interface can
be uninstalled using `pip uninstall oskarpy`

## Notes

- Both Python 2.7 and Python 3 are supported.
- Some examples showing how the Python interface can be used are shown
  in `examples/sim*` and `examples/spead`.
