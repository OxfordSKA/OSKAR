#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version $@
download_example_data $version

echo "Running OSKAR example beam pattern simulation"
echo "simulated visibilities"
echo ""
echo "  * OSKAR version          = $current_oskar_version"
echo "  * Example data version   = $version"
echo "  * Example data URL       = $example_data_url"
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd ${example_data_dir}

# Set or overwrite a number of settings in the example data settings file.
ini=setup.ini
# Telescope model
set_setting $ini telescope/input_directory telescope
# Observation parameters
set_setting $ini observation/num_channels 1
set_setting $ini observation/num_time_steps 10
# Beam pattern type options
set_setting $ini beam_pattern/coordinate_type 'Beam image'
set_setting $ini beam_pattern/coordinate_frame 'Equatorial'
set_setting $ini beam_pattern/beam_image/size 256
set_setting $ini beam_pattern/beam_image/fov_deg 180.0
# Output options
set_setting $ini beam_pattern/root_path example_beam_pattern
set_setting $ini beam_pattern/fits_file/save_voltage true
set_setting $ini beam_pattern/fits_file/save_total_intensity true
set_setting $ini beam_pattern/oskar_image_file/save_voltage  false
set_setting $ini beam_pattern/oskar_image_file/save_total_intensity false

# Run the interferometry simulation
run_beam_pattern $ini

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo "" 
echo "Results can be found in the directory: "
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""

