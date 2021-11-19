#!/bin/bash

#
# Script to clean up an OSKAR build directory.
#
# WARNING: Must only be run from inside the top level of the build directory
#

# NOTE: This check inserts the CMAKE build directory into this script.
# If the build directory is a symbolic link this check will fail if used
# from the absolute path rather than the link directory.
if [[ ! "$PWD" == "@PROJECT_BINARY_DIR@" && "$1" != "--force" ]]; then
    echo "ERROR: This script MUST be run from the top level build directory."
    echo "     : build directory   : '@PROJECT_BINARY_DIR@'"
    echo "     : current directory : '$PWD'"
    exit 1
fi

# Remove files and directories created by CMake
rm -rf CMake*
rm -f  CPack*
rm -f  CTest*
rm -f  Makefile
rm -f  cmake_install.cmake
rm -rf Testing
rm -f  Info.plist
rm -f  install_manifest.txt
rm -rf _CPack_Packages

# Remove OSKAR source module folders
rm -rf apps
rm -rf cmake
rm -rf docs
rm -rf extern
rm -rf gui
rm -rf oskar

# Remove any example data (zip files or directories)
rm -rf OSKAR-*-Example-Data

# Remove OS X .DS_Store file
rm -f  .DS_Store

# Remove any stray log files
rm -f  *.log

# Remove install folders (WARNING: Use this option with EXTREME CARE!!)
# Note can only remove the contents of the directories not the directories
# themselves as they may contain non-OSKAR files.
if [ "$1" == "-uninstall" ]; then
    rm -rf @CMAKE_INSTALL_PREFIX@/@OSKAR_BIN_INSTALL_DIR@/oskar*
    rm -f  @CMAKE_INSTALL_PREFIX@/@OSKAR_LIB_INSTALL_DIR@/liboskar*
    rm -rf @CMAKE_INSTALL_PREFIX@/@OSKAR_INCLUDE_INSTALL_DIR@/oskar
fi
