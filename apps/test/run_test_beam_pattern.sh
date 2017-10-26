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

# Set settings for the BP test.
set_setting $app $ini simulator/double_precision false
set_setting $app $ini telescope/input_directory telescope.tm
set_setting $app $ini observation/num_channels 2
set_setting $app $ini observation/start_frequency_hz 100e6
set_setting $app $ini observation/frequency_inc_hz 100e6
set_setting $app $ini observation/num_time_steps 5
set_setting $app $ini observation/start_time_utc "01-01-2000 12:00:00.000"
set_setting $app $ini observation/length "12:00:00.000"

# Beam "image" options
group=beam_pattern
set_setting $app $ini $group/root_path beam
set_setting $app $ini $group/beam_image/size 256
set_setting $app $ini $group/beam_image/fov_deg 180.0
set_setting $app $ini $group/coordinate_frame Equatorial
set_setting $app $ini $group/coordinate_type "Beam image"

# Station selection
set_setting $app $ini $group/all_stations false
set_setting $app $ini $group/station_ids 0,1,2

# Averaging options
group=beam_pattern/output
set_setting $app $ini $group/separate_time_and_channel true
set_setting $app $ini $group/average_time_and_channel true
# allowed values = None, Time, or Channel
set_setting $app $ini $group/average_single_axis Time

# Station outputs : Text file (voltage, amp, phase, power)
group=beam_pattern/station_outputs/text_file
set_setting $app $ini $group/raw_complex true
set_setting $app $ini $group/amp true
set_setting $app $ini $group/phase true
set_setting $app $ini $group/auto_power true

# Station outputs : FITS (amp, phase, power)
group=beam_pattern/station_outputs/fits_image
set_setting $app $ini $group/amp true
set_setting $app $ini $group/phase true
set_setting $app $ini $group/auto_power true

# Telescope (Interferometer) outputs : Text file (complex, amp, phase)
group=beam_pattern/telescope_outputs/text_file
set_setting $app $ini $group/cross_power_raw_complex true
set_setting $app $ini $group/cross_power_amp true
set_setting $app $ini $group/cross_power_phase true

# Telescope (Interferometer) outputs : FITS (amp, phase)
group=beam_pattern/telescope_outputs/fits_image
set_setting $app $ini $group/cross_power_amp true
set_setting $app $ini $group/cross_power_phase true

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
echo "This produced the following ${#beams[*]} beam pattern files:"
echo ""
idx=1
for beam in ${beams[*]}; do
    echo "  ${idx}. ${beam}"
    idx=$((idx+1))
done
echo ""
echo "Which can be found in the output directory:"
echo ""
echo "  ${example_data_dir}"
echo "-------------------------------------------------------------------------"
echo ""
