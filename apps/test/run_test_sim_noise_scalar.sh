#!/bin/bash

###############################################################################
#
# Description:
#   Tests the simulation of noise when producing visibilities.
#
# Method:
#   1. Generate an OSKAR visibility binary file with no sources with system
#      noise simulation enabled.
#   3. Report the noise on the Stokes-I visibilities.
#   4. Report the noise on an image made using the oskar_imager application.
#
###############################################################################

source @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

run_dir=$PWD
get_example_data_version "$@"
download_example_data "$version"

echo "Running OSKAR example demonstrating noise addition"
echo ""
echo "  * OSKAR version          = $current_oskar_version"
echo "  * Example data version   = $version"
echo "  * Example data URL       = $example_data_url"
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

ini=setup.ini
del_setting $ini sky/oskar_sky_model/file
set_setting $ini simulator/keep_log_file false
set_setting $ini simulator/write_status_to_log_file true
set_setting $ini simulator/double_precision false
set_setting $ini telescope/input_directory telescope.tm
set_setting $ini telescope/normalise_beams_at_phase_centre false
set_setting $ini telescope/pol_mode "Scalar"
#set_setting $ini telescope/pol_mode "Full"
set_setting $ini interferometer/max_time_samples_per_block 5
#set_setting $ini interferometer/ms_filename example.ms
set_setting $ini interferometer/correlation_type Both
set_setting $ini observation/num_channels 1
set_setting $ini observation/start_frequency_hz 200e6
set_setting $ini observation/num_time_steps 20
set_setting $ini observation/length "$(bc -l <<< 12.*3600.)"

# Set settings which add noise.
noise_seed=$((1 + RANDOM % 1000000))
rms=5
Tacc=10.0
bw=10.0
set_setting $ini interferometer/channel_bandwidth_hz $bw
set_setting $ini interferometer/time_average_sec $Tacc
set_setting $ini interferometer/noise/enable true
set_setting $ini interferometer/noise/seed $noise_seed
set_setting $ini interferometer/noise/freq "Observation settings"
set_setting $ini interferometer/noise/rms "Range"
set_setting $ini interferometer/noise/rms/start $rms
set_setting $ini interferometer/noise/rms/end $rms

# Run the interferometry simulation
echo "Starting interferometry simulation (with no sources)"
T0="$(date +%s)"
run_sim_interferometer -q $ini # &> /dev/null
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""


###############################################################################
noise_vis=example.vis

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
expected_XX_rms=$(bc -l <<< "sqrt(2)*$rms")
expected_I_rms=$(bc -l <<< "$expected_XX_rms/sqrt(2)")
expected_XX_mean=$(bc -l <<< "$expected_XX_rms*sqrt(2.0*$bw*$Tacc)")
echo "+ Channel bandwidth                : $bw Hz"
echo "+ Correlator dump time             : $Tacc s"
echo "+ Expected Linear polarisation STD : $(printf "%.3f Jy" "$expected_XX_rms")"
echo "+ Expected Stokes STD              : $(printf "%.3f Jy" "$expected_I_rms")"
echo "+ Expected cross-correlation mean  : 0+i0 Jy"
echo "+ Expected auto-correlation mean   : $(printf "%.3f Jy" "$expected_XX_mean")"
echo ""
echo ""

###############################################################################

# Image noisy visibilities
set_setting $ini image/image_type "I"
set_setting $ini image/fov_deg 5
set_setting $ini image/size 1024
set_setting $ini image/time_snapshots false
set_setting $ini image/input_vis_data $noise_vis
set_setting $ini image/root_path "with_noise"
echo "Imaging"
T0="$(date +%s)"
run_imager -q $ini
echo "  Finished in $(($(date +%s)-T0)) s"
echo ""

num_stations=30
noise_rms=$(get_setting $ini interferometer/noise/rms/start)
num_times=$(get_setting $ini observation/num_time_steps)
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
echo "  '$run_dir/$example_data_dir'"
echo "-------------------------------------------------------------------------"
echo ""
