#!/bin/bash
. @OSKAR_BINARY_DIR@/apps/test/test_utility.sh

function run_example() {
    local cmd=@OSKAR_BINARY_DIR@/apps/test/$1.sh
    eval "$cmd $version" # > /dev/null
    if [ $? != 0 ]; then
        echo "ERROR: Failed to run example: $1"
        exit 1
    fi
}

# Examples
run_example run_example_beam_pattern_equatorial
run_example run_example_interferometry

# Tests
# run_example run_test_add_noise_scalar
# run_example run_test_sim_noise_scalar
# run_example run_test_beam_pattern

# Benchmarks
#run_example run_benchmark_sim_interferometer
