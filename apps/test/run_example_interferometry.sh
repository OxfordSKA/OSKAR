#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

echo "Running OSKAR example: Interferometry simulation and imaging."
echo ""
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}" || exit

# Set settings as specified in the OSKAR Example document section 3.1
app_sim=${oskar_app_path}/oskar_sim_interferometer
ini_sim=oskar_sim_interferometer.ini
set_setting $app_sim $ini_sim simulator/max_sources_per_chunk 1024
set_setting $app_sim $ini_sim sky/oskar_sky_model/file sky.osm
set_setting $app_sim $ini_sim telescope/input_directory telescope.tm

# Run the interferometry simulation
echo "Starting interferometry simulation"
T0="$(date +%s)"
# run_sim_interferometer -q $ini_sim
run_sim_interferometer $ini_sim
echo " - Finished in $(($(date +%s)-T0)) s"
echo ""

# Make images
ini_img=oskar_imager.ini
echo "Imaging interferometry simulation output"
T0="$(date +%s)"
run_imager -q $ini_img
echo " - Finished in $(($(date +%s)-T0)) s"

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo ""
echo "Results can be found in the directory: "
echo "  '$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
