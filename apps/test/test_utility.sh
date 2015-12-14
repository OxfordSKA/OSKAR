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
#       - exit_                                                               #
#   2. Functions for working with OSKAR example data.                         #
#       - example_data_script_usage                                           #
#       - get_example_data_version                                            #
#       - download_example_data                                               #
#       - set_oskar_binaries                                                  #
#   3. Functions for working with OSKAR.                                      #
#       - set_setting                                                         #
#       - get_setting                                                         #
#       - del_setting                                                         #
#       - run_sim_interferomer                                                #
#       - run_beam_pattern                                                    #
#       - run_imager                                                          #
#       - run_vis_add_noise                                                   #
#       - run_vis_stats                                                       #
#       - run_fits_image_stats                                                #
###############################################################################


###############################################################################
#                                                                             #
# 0. Global variables                                                         #
#                                                                             #
###############################################################################
export current_oskar_version="@OSKAR_VERSION_STR@"
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
    echo "[$(date +'%a %Y-%m-%d %H:%M:%S %z')]" "$1"
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
    read -p "$1 ($2): " NEW_VALUE
    if [ -z "$NEW_VALUE" ]; then
        NEW_VALUE=$2
    fi
}
# Description: Exits if not running an interactive shell
# Usage: exit_ 99
function exit_()
{
    if [ -z "${PS1+x}" ]; then
        exit "$1"
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
        exit_ 0
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
    if [ -f "$file" ]; then
        rm -f "$file"
    fi
    if [ -d "$example_data_dir" ]; then
        rm -rf "$example_data_dir"
    fi
    wget -q "$example_data_url"
    if [ ! -f "$file" ]; then
        echo "Error: Failed to download example data from:"
        echo "  '$example_data_url'"
        echo ""
        echo "Please check the example data for OSKAR version '${version}' exists!"
        example_data_script_usage
        exit_ 1
    fi
    unzip -q "${file}"
    if [ ! -d "$example_data_dir" ]; then
        echo "ERROR: Failed to unpack example data. ${file}"
        exit_ 1
    fi
    # Remove the zip file.
    if [ -f "$file" ]; then
        rm -f "$file"
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
        exit_ 1
    fi
    if [ ! $# -eq 3 ]; then
        echo "ERROR: set_setting requires 3 input arguments got $#."
        echo "usage: set_setting [ini_file] [key] [value]"
        exit_ 1
    fi
    eval "$bin -q $1 $2 \"$3\""
}

# Description:
#   Gets the specified setting into the specified *.ini file using the
#   'oskar_get_setting' binary.
#
# Usage:
#   get_setting [ini_file] [key]
#
# Example
#   value=$(get_setting test.ini observation/num_time_steps)
#
function get_setting() {
    local bin=${oskar_app_path}/oskar_settings_get
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
        return
    fi
    if [ ! $# -eq 2 ]; then
        echo "ERROR: get_setting requires 2 input arguments got $#."
        echo "usage: get_setting [ini_file] [key]"
        exit_ 1
        return
    fi
    var=$($bin "$1" "$2")
    echo "$var"
}

# Description:
#   Removes / deletes the specified setting in the specified *.ini file using the
#   'oskar_set_setting' binary.
#
# Usage:
#   del_setting [ini_file] [key]
#
# Example
#   del_setting test.ini sky/oskar_sky_model/file
#
function del_setting() {
    local bin=${oskar_app_path}/oskar_settings_set
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    if [ ! $# -eq 2 ]; then
        echo "ERROR: del_setting requires 2 input arguments got $#."
        echo "usage: del_setting [ini_file] [key]"
        exit_ 1
    fi
    cmd="$bin -q $1 $2"
    ($cmd)
}



# Description:
#   Runs the specified OSKAR binary.
#
# Usage:
#   run_oskar_bin <name> [command line args]
function run_oskar_bin() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local LAST=$((NARGS-1))
    local OPTIONS=("${ARGS[@]:1:${NARGS}}")
    local NOPTIONS=${#OPTIONS[@]}
    local name=${ARGS[0]}
    local bin=${oskar_app_path}/${name}
    if [ ! -x "$bin" ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    echo "--------------------------------------------------------------------"
    echo "  * name  : $name"
    echo "  * bin   : $bin"
    echo "  * NARGS : $NOPTIONS"
    echo "  * ARGS  : ${OPTIONS[*]}"
    echo "--------------------------------------------------------------------"
}

# Description:
#   Runs the 'oskar_sim_interferometer' binary using the specified settings
#   file.
#
# Usage:
#   run_sim_interferomter [OPTONS] <ini_file>
#
# Example
#   run_sim_interferomter -q test.ini
#
function run_sim_interferometer() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local LAST=$((NARGS-1))
    local INI=${ARGS[${LAST}]}
    local OPTIONS=("${ARGS[@]:0:${LAST}}")
    local name="run_sim_interferometer"
    local bin=${oskar_app_path}/oskar_sim_interferometer
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    if [ ! -f "$INI" ]; then
        echo "ERROR: $name. Specified INI file not found!"
        echo "       INI file = '$INI'"
        echo ""
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    if [ "$NARGS" -lt 1 ]; then
        echo "ERROR: $name requires at least 1 input argument(s), got $#."
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    cmd="$bin ${OPTIONS[*]} $INI"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
    fi
}

# Description:
#   Runs the 'oskar_sim_beam_pattern' binary using the specified settings
#   file.
#
# Usage:
#   run_beam_pattern [OPTIONS] [ini_file]
#
# Example
#   run_beam_pattern test.ini
#
function run_beam_pattern() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local LAST=$((NARGS-1))
    local INI=${ARGS[${LAST}]}
    local OPTIONS=("${ARGS[@]:0:${LAST}}")
    local name="oskar_sim_beam_pattern"
    local bin=${oskar_app_path}/oskar_sim_beam_pattern
    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    if [ ! -f "$INI" ]; then
        echo "ERROR: $name. Specified INI file not found!"
        echo "       INI file = '$INI'"
        echo ""
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    if [ "$NARGS" -lt 1 ]; then
        echo "ERROR: $name requires at least 1 input argument(s), got $#."
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    cmd="$bin ${OPTIONS[*]} $INI"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
    fi
}

# Description:
#   Runs the 'oskar_imager' binary using the specified settings
#   file.
#
# Usage:
#   run_imager [OPTIONS] <ini_file>
#
# Example
#   run_imager -q test.ini
#
function run_imager() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local LAST=$((NARGS-1))
    local INI=${ARGS[${LAST}]}
    local OPTIONS=("${ARGS[@]:0:${LAST}}")
    local name="run_imager"
    local bin=${oskar_app_path}/oskar_imager
    if [ ! -x "$bin" ]; then
        echo "ERROR: $1 unable to find required binary: $bin."
        exit_ 1
    fi
    if [ ! -f "$INI" ]; then
        echo "ERROR: $name. Specified INI file not found!"
        echo "       INI file = '$INI'"
        echo ""
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    if [ "$NARGS" -lt 1 ]; then
        echo "ERROR: $name requires at least 1 input argument(s), got $#."
        echo "usage: $name [OPTIONS] <ini_file>"
        exit_ 1
    fi
    cmd="$bin ${OPTIONS[*]} $INI"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
    fi
}

# Description:
#   Runs the 'oskar_vis_add_noise' binary with the specified settings
#   file and visibility binary data file
#
# Usage:
#   run_vis_add_noise [OPTIONS] <ini_file> <vis file>
#
# Example
#   run_vis_add_noise -q test.ini test.vis
#   run_vis_add_noise -v test.ini test.vis
#
function run_vis_add_noise() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local IVIS=$((NARGS-1)) # Index of last argument
    local VIS=${ARGS[${IVIS}]}
    local IINI=$((NARGS-2)) # Index of 2nd to last argument
    local INI=${ARGS[${IINI}]}
    local OPTIONS=("${ARGS[@]:0:${IINI}}") # Options
    local name="oskar_vis_add_noise"
    local bin=${oskar_app_path}/oskar_vis_add_noise
    if [ ! -x "$bin" ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    if [ ! -f "$INI" ]; then
        echo "ERROR: $name. Specified INI file not found!"
        echo "       INI file = '$INI'"
        echo ""
        echo "usage: $name [OPTIONS] <ini_file> <vis file>"
        exit_ 1
    fi
    if [ ! -f "$VIS" ]; then
        echo "ERROR: $name. Specified VIS file not found!"
        echo "       VIS file = '$VIS'"
        echo ""
        echo "usage: $name [OPTIONS] <ini_file> <vis file>"
        exit_ 1
    fi
    if [ "$NARGS" -lt 2 ]; then
        echo "ERROR: $name requires at least 2 input argument(s), got $#."
        echo "usage: $name [OPTIONS] <ini_file> <vis file>"
        exit_ 1
    fi
    cmd="$bin ${OPTIONS[*]} -s $INI $VIS"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
    fi
}


# Description:
#   Runs the 'oskar_vis_summary' binary with the specified visibility binary
#   data file
#
# Usage:
#   run_vis_stats [OPTIONS] <vis file>
#
# Example
#   run_vis_stats --stats test.vis
#
function run_vis_stats() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local name="oskar_vis_summary"
    local bin=${oskar_app_path}/${name}
    if [ ! -x "$bin" ]; then
        echo "ERROR: Unable to find required binary: $bin."
        exit_ 1
    fi
    if [ "$NARGS" -lt "1" ]; then
        echo "ERROR: $name requires at least 1 input argument, got $#."
        echo "usage: $name [OPTIONS] <vis file(s)>"
        exit_ 1
        return
    fi
    cmd="$bin --stats ${ARGS[*]}"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
        return
    fi
}

# Description:
#   Runs the 'fits_image_stats' script with the specified FITS image
#
# Usage:
#   run_fits_image_stats [OPTIONS] <FITS image>
#
# Example
#   run_fits_image_stats -v test.fits
#
function run_fits_image_stats() {
    local ARGS=("$@")
    local NARGS=${#ARGS[@]}
    local name="fits_image_stats.py"
    local bin=${oskar_app_path}/test/${name}

    if [ ! -x ${bin} ]; then
        echo "ERROR: Unable to find required binary: $bin."
        return
    fi
    if [ "$NARGS" -lt "1" ]; then
        echo "ERROR: $name requires at least 1 input argument, got $#."
        echo "Usage: $name [OPTIONS] <FITS image>"
        exit_ 1
        return
    fi
    cmd="python $bin ${ARGS[*]}"
    ($cmd)
    if [ $? != 0 ]; then
        echo "ERROR: $name. Failed."
        exit_ 1
        return
    fi
}
