#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version $@
download_example_data $version

echo "Running OSKAR example demonstrating noise addition"
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
set_setting $ini telescope/normalise_beams_at_phase_centre false

# Run the interferometry simulation
echo "Starting interferometry simulation" 
T0="$(date +%s)"
run_sim_interferometer -q $ini
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""

# Set settings which add noise.
set_setting $ini interferometer/noise/enable true
set_setting $ini interferometer/noise/seed time
set_setting $ini interferometer/noise/freq 'Observation settings'
set_setting $ini interferometer/noise/values 'RMS flux density'
set_setting $ini interferometer/noise/values/rms Range
set_setting $ini interferometer/noise/values/rms/start 5
set_setting $ini interferometer/noise/values/rms/end 0.5

# Add noise to the visibilities.
echo "Adding noise to visibilities"
T0="$(date +%s)" 
vis=example.vis
run_vis_add_noise -q $ini $vis
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""

# Image visibilities with and without noise.
echo "Imaging visibilities"
T0="$(date +%s)" 
set_setting $ini image/input_vis_data example.vis
set_setting $ini image/root_path no_noise
run_imager -q $ini
set_setting $ini image/input_vis_data example_noise.vis
set_setting $ini image/root_path with_noise
run_imager -q $ini
echo "  Finished in $(($(date +%s)-T0)) s"

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo "" 
echo "Results can be found in the directory: "
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""

