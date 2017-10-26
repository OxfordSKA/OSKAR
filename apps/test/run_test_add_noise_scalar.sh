#!/bin/bash

###############################################################################
#
# Description:
#   Tests the addition of noise to visibilities.
#
# Method:
#   1. Generate an OSKAR visibility binary file with no sources. This has
#      visibility amplitudes which are all zero.
#   2. Add noise using the oskar_vis_add_noise application.
#   3. Report the noise on the Stokes-I visibilities.
#   4. Report the noise on an image made using the oskar_imager application.
#
###############################################################################

source @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

echo "Running OSKAR example demonstrating noise addition"
echo ""
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

app_sim=${oskar_app_path}/oskar_sim_interferometer
ini_sim=oskar_sim_interferometer.ini
del_setting $app_sim $ini_sim sky/oskar_sky_model/file
set_setting $app_sim $ini_sim simulator/keep_log_file false
set_setting $app_sim $ini_sim simulator/write_status_to_log_file true
set_setting $app_sim $ini_sim simulator/double_precision false
set_setting $app_sim $ini_sim telescope/input_directory telescope.tm
set_setting $app_sim $ini_sim telescope/normalise_beams_at_phase_centre false
set_setting $app_sim $ini_sim telescope/pol_mode "Scalar"
#set_setting $app_sim $ini_sim telescope/pol_mode "Full"
set_setting $app_sim $ini_sim observation/num_channels 1
set_setting $app_sim $ini_sim observation/start_frequency_hz 200e6
set_setting $app_sim $ini_sim observation/num_time_steps 20
set_setting $app_sim $ini_sim observation/length "$(bc -l <<< 12.*3600.)"

# Run the interferometry simulation
echo "Starting interferometry simulation (with no sources)"
T0="$(date +%s)"
run_sim_interferometer -q $ini_sim # &> /dev/null
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""


###############################################################################


# Set settings which add noise.
noise_seed=$((1 + RANDOM % 1000000))
rms=5
set_setting $app_sim $ini_sim interferometer/noise/enable true
set_setting $app_sim $ini_sim interferometer/noise/seed $noise_seed
set_setting $app_sim $ini_sim interferometer/noise/freq "Observation settings"
set_setting $app_sim $ini_sim interferometer/noise/rms "Range"
set_setting $app_sim $ini_sim interferometer/noise/rms/start $rms
set_setting $app_sim $ini_sim interferometer/noise/rms/end $rms

# Add noise to the visibilities.
echo "Adding noise to visibilities, RMS specification."
echo "  - RMS         : $(printf "%.2e" "$rms") Jy"
T0="$(date +%s)"
vis=example.vis
noise_vis=example_noise.vis
run_vis_add_noise -q $ini_sim $vis
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""

# Print some stats from the visibilities after noise addition
echo "Visibility stats [$noise_vis]"
echo "........................................................................."
run_vis_stats $noise_vis > stats.txt
lines=$(wc -l < stats.txt)
sed -n "7,$((lines-3))p" "stats.txt"
echo "........................................................................."
echo ""
echo ""
#
# Expect RMS to be sqrt(2) larger in single polarisation measurements as
# real and imag parts of the complex correlator output are uncorrelated.
# Expect the RMS of Stokes parameters which combine pairs of polarisations
# to be sqrt(2) smaller than the linear polarisation.
#
expected_1pol_rms=$(bc -l <<< "sqrt(2)*$rms")
echo "+ Expected Linear polarisation RMS : $(printf "%.3f" "$expected_1pol_rms")"
echo "+ Expected Stokes RMS              : $(printf "%.3f" "$rms")"
echo ""
echo ""

###############################################################################

# Image noisy visibilities
app_img=${oskar_app_path}/oskar_imager
ini_img=oskar_imager.ini
set_setting $app_img $ini_img image/image_type "I"
set_setting $app_img $ini_img image/fov_deg 5
set_setting $app_img $ini_img image/size 1024
set_setting $app_img $ini_img image/time_snapshots false
set_setting $app_img $ini_img image/input_vis_data $noise_vis
set_setting $app_img $ini_img image/root_path "with_noise"
echo "Imaging"
T0="$(date +%s)"
run_imager -q $ini_img
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""

num_stations=30
noise_rms=$(get_setting $app_sim $ini_sim interferometer/noise/rms/start)
num_times=$(get_setting $app_sim $ini_sim observation/num_time_steps)
num_baselines=$((num_stations*(num_stations-1)/2))
num_samples=$((num_baselines*num_times))
#
# Noise reduction factors:
#   - Single polarisation to Stokes-I  = 1/sqrt(2)
#   - Number of independent samples    = 1/sqrt(num_baselines*num_times)
#
expected_std=$(bc -l <<< "$noise_rms/sqrt(2*$num_baselines*$num_times)")
img_std=$(run_fits_image_stats with_noise_I.fits)
std_diff=$(bc -l <<< "$img_std-$expected_std")

echo "-------------------------------------------------------------------------"
echo "Image noise:"
echo "  + No. stations          : $num_stations"
echo "  + No. times             : $num_times"
echo "  + No. baselines         : $num_baselines"
echo "  + No. visibilities      : $num_samples  (${num_times} x ${num_baselines})"
echo "  + Visibility noise      : $(printf "%.3f" "$noise_rms") Jy"
echo "  + Image noise reduction : 1/sqrt(2*$num_samples)"
echo "  + Expected image noise  : $(printf "%.5f" "$expected_std") Jy/beam"
echo "  + Measured image noise  : $(printf "%.5f" "$img_std") Jy/beam"
echo "  + Difference            : $(printf "%e" "$std_diff") Jy/beam"
echo "-------------------------------------------------------------------------"

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo ""
echo "Results can be found in the directory: "
echo "  '$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
