#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version "$@"
download_example_data "$version"

echo "Running OSKAR example beam pattern simulation"
echo "simulated visibilities"
echo ""
echo "  * OSKAR version          = $current_oskar_version"
echo "  * Example data version   = $version"
echo "  * Example data URL       = $example_data_url"
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

# Set or overwrite a number of settings in the example data settings file.
ini=setup.ini
# Telescope model
set_setting $ini telescope/input_directory telescope.tm

# Run the beam pattern simulation
echo "Running beam pattern simulation"
T0="$(date +%s)"
run_beam_pattern -q $ini
echo "  Finished in $(($(date +%s)-T0)) s"

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo ""
echo "Results can be found in the directory: "
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
