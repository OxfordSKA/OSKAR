#!/bin/bash
source @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

# Optional --verbose option to not hide the oskar run log.
if [ "$2" == "--verbose" ]; then
    verbose=''
else
    verbose='-q'
fi

echo "Running OSKAR example: Interferometry benchmark simulation."
echo ""
echo "  * Example data directory = $example_data_dir"
echo ""

# Move into the example data directory
cd "${example_data_dir}"

# Set settings
app=${oskar_app_path}/oskar_sim_interferometer
ini=oskar_sim_interferometer.ini
del_setting $app $ini sky/oskar_sky_model/file
set_setting $app $ini telescope/input_directory telescope.tm
set_setting $app $ini simulator/keep_log_file true
set_setting $app $ini simulator/write_status_to_log_file true
set_setting $app $ini observation/num_channels 1
set_setting $app $ini observation/num_time_steps 30
set_setting $app $ini sky/generator/grid/side_length 64
set_setting $app $ini sky/generator/grid/fov_deg 5
set_setting $app $ini sky/generator/grid/mean_flux_jy 1

# Run the interferometry simulation in single precision
set_setting $app $ini simulator/double_precision false
echo "Starting interferometry simulation [Single precision]"
T0="$(date +%s)"
run_sim_interferometer $verbose $ini
echo " - Finished in ~$(($(date +%s)-T0)) s"
oskar_log=$(ls oskar*.log)
mv "${oskar_log}" "SINGLE_${oskar_log}"
echo "........................................................................."
sed -n '125,136p' "SINGLE_${oskar_log}"
echo "........................................................................."
echo ""

# Run the interferometry simulation in double precision
set_setting $app $ini simulator/double_precision true
echo "Starting interferometry simulation [Double precision]"
T0="$(date +%s)"
run_sim_interferometer $verbose $ini
echo " - Finished in ~$(($(date +%s)-T0)) s"
oskar_log=$(ls oskar*.log)
mv "$oskar_log" "DOUBLE_${oskar_log}"
echo "........................................................................."
sed -n '125,136p' "DOUBLE_${oskar_log}"
echo "........................................................................."
echo ""

echo ""
echo "-------------------------------------------------------------------------"
echo "Run complete!"
echo ""
echo "Please inspect the OSKAR run logs for timing results. This benchmark has"
echo "generated the following log files:"
echo ""
oskar_logs=$(ls ./*.log)
idx=1
for log in ${oskar_logs[*]}; do
    echo "  ${idx}. ${log}"
    idx=$((idx+1))
done
echo ""
echo "Which can be found in the output directory:"
echo "  ${example_data_dir}"
echo "-------------------------------------------------------------------------"
echo ""
