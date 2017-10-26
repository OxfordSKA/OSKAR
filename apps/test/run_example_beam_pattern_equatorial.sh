#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

echo "Running OSKAR example beam pattern simulation"
echo "simulated visibilities"
echo ""
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

# Set or overwrite a number of settings in the example data settings file.
app=${oskar_app_path}/oskar_sim_beam_pattern
ini=oskar_sim_beam_pattern.ini
# Telescope model
set_setting $app $ini telescope/input_directory telescope.tm

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
echo "  '$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
