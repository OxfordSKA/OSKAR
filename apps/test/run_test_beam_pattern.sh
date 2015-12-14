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

# Remove setting that are not used in BP mode.
del_setting $ini interferometer/oskar_vis_filename
del_setting $ini interferometer/channel_bandwidth_hz
del_setting $ini interferometer/time_average_sec
del_setting $ini sky/oskar_sky_model/file
del_setting $ini image/fov_deg
del_setting $ini image/size
del_setting $ini image/image_type
del_setting $ini image/time_snapshots
del_setting $ini image/input_vis_data
del_setting $ini image/root_path
del_setting $ini image/fits_image

# Set settings for the BP test.
set_setting $ini simulator/double_precision false
set_setting $ini telescope/input_directory telescope.tm
set_setting $ini observation/num_channels 2
set_setting $ini observation/start_frequency_hz 100e6
set_setting $ini observation/frequency_inc_hz 100e6
set_setting $ini observation/num_time_steps 5
set_setting $ini observation/start_time_utc "01-01-2000 12:00:00.000"
set_setting $ini observation/length "12:00:00.000"

# Beam "image" options
group=beam_pattern
set_setting $ini $group/root_path beam
set_setting $ini $group/size 256
set_setting $ini $group/fov_deg 180.0
set_setting $ini $group/coordinate_frame Equatorial
set_setting $ini $group/coordinate_type "Beam image"

# Station selection
set_setting $ini $group/all_stations false
set_setting $ini $group/station_ids 0,1,2

# Averaging options
group=beam_pattern/output
set_setting $ini $group/separate_time_and_channel true
set_setting $ini $group/average_time_and_channel true
# allowed values = None, Time, or Channel
set_setting $ini $group/average_single_axis Time

# Station outputs : Text file (voltage, amp, phase, power)
group=beam_pattern/station_outputs/text_file
set_setting $ini $group/raw_complex true
set_setting $ini $group/amp true
set_setting $ini $group/phase true
set_setting $ini $group/auto_power_stokes_i true

# Station outputs : FITS (amp, phase, power)
group=beam_pattern/station_outputs/fits_image
set_setting $ini $group/amp true
set_setting $ini $group/phase true
set_setting $ini $group/auto_power_stokes_i true

# Telescope (Interferometer) outputs : Text file (complex, amp, phase)
group=beam_pattern/telescope_outputs/text_file
set_setting $ini $group/cross_power_stokes_i_raw_complex true
set_setting $ini $group/cross_power_stokes_i_amp true
set_setting $ini $group/cross_power_stokes_i_phase true

# Telescope (Interferometer) outputs : FITS (amp, phase)
group=beam_pattern/telescope_outputs/fits_image
set_setting $ini $group/cross_power_stokes_i_amp true
set_setting $ini $group/cross_power_stokes_i_phase true

# Run the beam pattern simulation
echo "Running beam pattern simulation"
T0="$(date +%s)"
#run_beam_pattern -q $ini
run_beam_pattern $ini
echo "  Finished in $(($(date +%s)-T0)) s"

echo ""
echo "-------------------------------------------------------------------------"
beams=(./*.fits)
echo "Run complete!"
echo ""
echo "This produced the following beam ${#beams[*]} pattern files:"
echo ""
idx=1
for beam in ${beams[*]}; do
    echo "  ${idx}. ${beam}"
    idx=$((idx+1))
done
echo ""
echo "Which can be found in the output directory:"
echo ""
echo "  ${run_dir}/${example_data_dir}"
echo "-------------------------------------------------------------------------"
echo ""
