#!/bin/bash

# Script variables (Variables enclosed in '@' symbols are set by CMake)
# ----------------------------------------------------------------------------
app_path="@PROJECT_BINARY_DIR@/apps"
default_example_version="@OSKAR_VERSION_MAJOR@.@OSKAR_VERSION_MINOR@"
current_oskar_version="@OSKAR_VERSION_STR@"
oskar_url="http://oerc.ox.ac.uk/~ska/oskar"
oskar_bin_set=${app_path}/oskar_settings_set
oskar_bin_sim=${app_path}/oskar_sim_interferometer
oskar_bin_img=${app_path}/oskar_imager
ini_file=setup.ini
# ----------------------------------------------------------------------------

# Parse command line arguments.
if [ $# -eq 1 ]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [OSKAR example data version (major.minor)]"
        echo "Example: $0 2.5"
        exit 0
    fi
    version=$1
else
    version=$default_example_version
fi

# Check required OSKAR binaries can be found are are executable
if [ ! -x ${oskar_bin_set} ]; then
    echo "ERROR: Unable to find required binary: $oskar_bin_set."
    exit 1
fi
if [ ! -x ${oskar_bin_sim} ]; then
    echo "ERROR: Unable to find required binary: $oskar_bin_sim."
    exit 1
fi
if [ ! -x ${oskar_bin_img} ]; then
    echo "ERROR: Unable to find required binary: $oskar_bin_img."
    exit 1
fi

# Set variables to the example data path, file, and url
path="OSKAR-${version}-Example-Data"
file="${path}.zip"
url="${oskar_url}/${version}/data/${file}"
run_dir=$PWD # Directory from which the script was run.

echo "Running OSKAR examples 3.3 & 3.4: Interferometry Simulation and Imaging"
echo "simulated visibilities"
echo ""
echo "  * Application path = '$app_path'"
echo "  * OSKAR version    = $current_oskar_version"
echo "  * Example version  = $version"
echo "  * Example data     = $url"
echo ""

# Download and unpack the example data, removing any existing data first.
if [ -f $file ]; then
    rm -f $file
fi
if [ -d $path ]; then
    rm -rf $path
fi
wget -q $url
if [ ! -f $file ]; then
    echo "ERROR: Failed to dowload example data. Please check the example data"
    echo "       for OSKAR ${version} exists!"
    exit 1
fi
unzip -q ${file}
if [ ! -d $path ]; then
    echo "ERROR: Failed to unpack example data."
    exit 1
fi

# Move into the example data directory
cd ${path}

# Set settings as specified in the OSKAR Example document section 3.1
eval "${oskar_bin_set} -q ${ini_file} sky/oskar_sky_model/file sky.osm"
eval "${oskar_bin_set} -q ${ini_file} telescope/input_directory telescope"

# Run the interferometry simulation
eval "${oskar_bin_sim} ${ini_file}"

# Make oskar images
eval "${oskar_bin_img} ${ini_file}"

# Remove the example data Zip file.
cd $run_dir
if [ -f $file ]; then
    rm -f $file
fi

