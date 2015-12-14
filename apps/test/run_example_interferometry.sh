#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version "$@"
download_example_data "$version"

echo "Running OSKAR example: Interferometry simulation and DFT imaging."
echo ""
echo "  * OSKAR version          = $current_oskar_version"
echo "  * Example data version   = $version"
echo "  * Example data URL       = $example_data_url"
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

# Set settings as specified in the OSKAR Example document section 3.1
ini=setup.ini
set_setting $ini sky/oskar_sky_model/file sky.osm
set_setting $ini telescope/input_directory telescope.tm

# Run the interferometry simulation
echo "Starting interferometry simulation"
T0="$(date +%s)"
run_sim_interferometer -q $ini
echo " - Finished in $(($(date +%s)-T0)) s"
echo ""

# Make images
echo "Imaging interferometry simulation output"
T0="$(date +%s)"
run_imager -q $ini
echo " - Finished in $(($(date +%s)-T0)) s"

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo ""
echo "Results can be found in the directory: "
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
