#!/bin/bash

#
# Script clean up an OSKAR build directory.
#
# WARNING: Must only be run from inside the top level of the build directory
#

# NOTE: This check inserts the CMAKE build directory into this script. 
# If the build directory is a symbolic link this check will fail if used
# from the absolute path rather than the link directory.
if [ ! "$PWD" == "@PROJECT_BINARY_DIR@" ]; then
    echo "ERROR: This script MUST be run from the top level build directory."
    exit 1
fi

# Remove files and directories created by CMake
rm -rf CMake*
rm -f  CPack*
rm -f  CTest*
rm -f  Makefile
rm -f  cmake_install.cmake
rm -rf Testing
rm -f install_manifest.txt

# Remove OSKAR source module folders
rm -rf apps
rm -rf convert
rm -rf correlate
rm -rf element
rm -rf extern
rm -rf fits
rm -rf imaging
rm -rf interferometry
rm -rf jones
rm -rf math
rm -rf matlab
rm -rf ms
rm -rf python
rm -rf settings
rm -rf sky
rm -rf splines
rm -rf station
rm -rf utility

# Remove the main OSKAR library
rm -f  liboskar*

# Remove any example data (zip files or directories)
rm -rf OSKAR-*-Example-Data

# Remove OS X .DS_Store file
rm -f  .DS_Store

# Remove any stray log files
rm -f  *.log

# Remove install folders (WARNING: Use this option with CARE!!)
if [ "$1" == "-uninstall" ]; then
    rm -f  @CMAKE_INSTALL_PREFIX@/@OSKAR_BIN_INSTALL_DIR@/oskar*
    rm -f  @CMAKE_INSTALL_PREFIX@/@OSKAR_LIB_INSTALL_DIR@/liboskar*
    rm -rf @CMAKE_INSTALL_PREFIX@/@OSKAR_INCLUDE_INSTALL_DIR@
    rm -rf @CMAKE_INSTALL_PREFIX@/@OSKAR_MATLAB_INSTALL_DIR@
    rm -rf @CMAKE_INSTALL_PREFIX@/@OSKAR_PYTHON_INSTALL_DIR@
fi

