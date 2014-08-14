#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version $@
download_example_data $version

echo "Running OSKAR example demonstrating noise addition"
echo "simulated visibilities"
echo ""
echo "  * OSKAR version          = $current_oskar_version"
echo "  * Example data version   = $version"
echo "  * Example data URL       = $example_data_url"
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd ${example_data_dir}

# Set settings as specified in the OSKAR Example document section 3.1
ini=setup.ini
set_setting $ini sky/oskar_sky_model/file sky.osm
set_setting $ini telescope/input_directory telescope

# Run the interferometry simulation
run_sim_interferometer $ini

# Make images
run_imager $ini

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo "" 
echo "Results can be found in the directory: "
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""

