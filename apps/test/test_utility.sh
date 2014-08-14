#!/bin/bash

###############################################################################
# Introduction                                                                #
###############################################################################
# This is a set of utility functions for writing OSKAR test scripts.          #
#                                                                             #
# Note some variables in this script (those surrounded by '@' symbols are     #
# replaced by CMake using the configure_file function.                        #
#                                                                             #
# Functions are grouped into the following sections:                          #
#   0. Global variables                                                       #
#       - current_oskar_version                                               #
#       - oskar_app_path                                                      #
#       - oskar_url                                                           #
#   1. General utility functions                                              #
#       - console_log                                                         #
#       - set_value                                                           #
#   2. Functions for working with OSKAR example data.                         #
#       - example_data_script_usage                                           #
#       - get_example_data_version                                            #
#       - download_example_data                                               #
#       - set_oskar_binaries                                                  #
#   3. Functions for working with OSKAR.                                      #
#       - set_setting                                                         #
#       - run_sim_interferomer                                                #
#       - run_beam_pattern                                                    #
#       - run_imager                                                          #
#       - run_vis_add_noise                                                   #
###############################################################################


###############################################################################
#                                                                             #
# 0. Global variables                                                         #
#                                                                             #
###############################################################################
current_oskar_version="@OSKAR_VERSION_STR@"
oskar_app_path="@PROJECT_BINARY_DIR@/apps"
oskar_url="http://oerc.ox.ac.uk/~ska/oskar"

###############################################################################
#                                                                             #
# 1. General utility functions                                                #
#                                                                             #
###############################################################################

# Description: Prints a time-stamped log message.
# Usage: console_log ['Hello World']
function console_log()
{
    echo '['$(date +'%a %Y-%m-%d %H:%M:%S %z')']' $1
}

# Description:
#   Prompts the user to enter some value with a default and saves the answer
#   into $NEW_VALUE
#
# Usage: 
#   set_value ['Enter something'] ['defaultValue']
#   VAR=$NEW_VALUE
function set_value()
{
    read -p "$1 ("$2"): " NEW_VALUE
    if [ -z $NEW_VALUE ]; then
        NEW_VALUE=$2
    fi
}

###############################################################################
#                                                                             #
# 2. Functions for working with OSKAR Example data                            #
#                                                                             #
###############################################################################

# Description:
#   Prints usage string for scripts working with example data.
#
# Usage:
#   example_data_script_usage
#
function example_data_script_usage() {
    echo ""
    echo "Usage:"   
    echo "  $0 [OSKAR example data version (major.minor)]"
    echo ""
    echo "Example:"
    echo "  $0 2.5"
    echo ""
}

# Description:
#   Extracts the OSKAR example data version from the command line and sets
#   it into the $version variable.
#
#   The example data version is comprised of the Major and Minor versions of
#   the OSKAR binaries for which the example was made.
#
#   ie. OSKAR 2.5.1 would have example version 2.5.
#
#   If no command line options are found, then the version defaults to the
#   current OSKAR binary version.    
#
# Usage:
#   getExampleVersion [Array of command line arguments]
#
# Example:
#   getExampleVersion $@
# 
function get_example_data_version() {
    local default_example_version="@OSKAR_VERSION_MAJOR@.@OSKAR_VERSION_MINOR@"
    # Parse command line arguments.
    if [[ $# -gt 1 || "$1" == "--help" || "$1" == "-h" || "$1" == "-?" ]]; then
        example_data_script_usage
        exit 0
    fi
    if [ $# -eq 1 ]; then
        version=$1
    else
        version=$default_example_version
    fi
}

# Description:
#   Downloads the OSKAR example data for the specified version.
#   Also sets the variables:
#       $example_data_url : The URL to the example data being used.
#       $example_data_dir : The directory of the example data downloaded
#
# Usage:
#   downloadExampleData [version]
#
# Example:
#   downloadExampleData $version
#   downloadExampleData 2.5
#
function download_example_data() {
    # Set variables to the example data path, file, and url
    example_data_dir="OSKAR-${1}-Example-Data"
    local file="${example_data_dir}.zip"
    example_data_url="${oskar_url}/${1}/data/${file}"
    # Download and unpack the example data, removing any existing data first.
    if [ -f $file ]; then 
        rm -f $file
    fi
    if [ -d $example_data_dir ]; then
        rm -rf $example_data_dir
    fi
    wget -q $example_data_url
    if [ ! -f $file ]; then
        echo "Error: Failed to download example data from:"
        echo "  '$example_data_url'"
        echo ""
        echo "Please check the example data for OSKAR ${version} exists!"
        example_data_script_usage
        exit 1
    fi
    unzip -q ${file}
    if [ ! -d $example_data_dir ]; then
        echo "ERROR: Failed to unpack example data. ${file}"
        exit 1
    fi
    # Remove the zip file.
    if [ -f $file ]; then
        rm -f $file
    fi
}

###############################################################################
#                                                                             #
# 3. Functions for working with OSKAR                                         #
#                                                                             #
###############################################################################

# Description:
#   Sets the specified setting into the specified *.ini file using the 
#   'oskar_set_setting' binary.
#
# Usage:
#   set_setting [ini_file] [key] [value]
#
# Example
#   set_setting test.ini sky/oskar_sky_model/file sky.osm 
#
function set_setting() {
    local bin=${oskar_app_path}/oskar_settings_set
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit 1
    fi
    if [ ! $# -eq 3 ]; then
        echo "ERROR: set_setting requires 3 input arguments got $#."
        echo "usage: set_setting [ini_file] [key] [value]"
        exit 1
    fi
    eval "${bin} -q $1 $2 $3"
}

# Description:
#   Runs the 'oskar_sim_interferometer' binary using the specified settings 
#   file.
#
# Usage:
#   run_sim_interferomter [ini_file]
#
# Example
#   run_sim_interferomter test.ini
#
function run_sim_interferometer() {
    local bin=${oskar_app_path}/oskar_sim_interferometer
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit 1
    fi
    if [ ! $# -eq 1 ]; then
        echo "ERROR: run_sim_interferomter requires 1 input arguments got $#."
        echo "usage: run_sim_interferomter [ini_file]"
        exit 1
    fi
    eval "${bin} $1"
}

# Description:
#   Runs the 'oskar_sim_beam_pattern' binary using the specified settings 
#   file.
#
# Usage:
#   run_beam_pattern [ini_file]
#
# Example
#   run_beam_pattern test.ini
#
function run_beam_pattern() {
    local bin=${oskar_app_path}/oskar_sim_beam_pattern
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit 1
    fi
    if [ ! $# -eq 1 ]; then
        echo "ERROR: run_beam_pattern requires 1 input arguments got $#."
        echo "usage: run_beam_pattern [ini_file]"
        exit 1
    fi
    eval "${bin} $1"
}

# Description:
#   Runs the 'oskar_imager' binary using the specified settings 
#   file.
#
# Usage:
#   run_imager [ini_file]
#
# Example
#   run_imager test.ini
#
function run_imager() {
    local bin=${oskar_app_path}/oskar_imager
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit 1
    fi
    if [ ! $# -eq 1 ]; then
        echo "ERROR: run_imager requires 1 input arguments got $#."
        echo "usage: run_imager [ini_file]"
        exit 1
    fi
    eval "${bin} $1"
}

# Description:
#   Runs the 'oskar_vis_add_noise' binary with the specified settings 
#   file and visibility binary data file
#
# Usage:
#   run_vis_add_noise [ini_file] [vis file]
#
# Example
#   run_vis_add_noise test.ini test.vis
#
function run_vis_add_noise() {
    local bin=${oskar_app_path}/oskar_vis_add_noise
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit 1
    fi
    if [ ! $# -eq 2 ]; then
        echo "ERROR: run_vis_add_noise requires 2 input arguments got $#."
        echo "usage: run_vis_add_noise [ini_file] [vis file]"
        exit 1
    fi
    eval "${bin} -v -s $1 $2"
}