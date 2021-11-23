/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_option_parser.h"
#include "correlate/oskar_cross_correlate.h"
#include "interferometer/oskar_jones.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "oskar_version.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static void benchmark(int num_stations, int num_sources, int type,
        int jones_type, int location, int use_extended,
        int use_bandwidth_smearing, int use_time_smearing,
        int niter, std::vector<double>& times, const std::string& ascii_file,
        int* status);

int main(int argc, char** argv)
{
    oskar::OptionParser opt("oskar_correlator_benchmark", OSKAR_VERSION_STR);
    opt.add_flag("-nst", "Number of stations.", 1, "", true);
    opt.add_flag("-nsrc", "Number of sources.", 1, "", true);
    opt.add_flag("-sp", "Use single precision (default: double precision)");
    opt.add_flag("-s", "Use scalar Jones terms (default: matrix/polarised).");
    opt.add_flag("-g", "Run on the GPU");
    opt.add_flag("-c", "Run on the CPU");
    opt.add_flag("-cl", "Run using OpenCL");
    opt.add_flag("-e", "Use Gaussian sources (default: point sources).");
    opt.add_flag("-b", "Use bandwidth smearing (default: no bandwidth smearing).");
    opt.add_flag("-t", "Use time smearing (default: no time smearing).");
    opt.add_flag("-r", "Dump raw iteration data to this file.", 1);
    opt.add_flag("-a", "Dump ASCII visibility data to this file.", 1);
    opt.add_flag("-std", "Discard values greater than this number of standard "
            "deviations from the mean.", 1);
    opt.add_flag("-n", "Number of iterations", 1, "1", false);
    opt.add_flag("-v", "Display verbose output.", false);
    if (!opt.check_options(argc, argv))
    {
        return EXIT_FAILURE;
    }

    int location = -1, status = 0;
    double max_std_dev = 0.0;
    int num_stations = opt.get_int("-nst");
    int num_sources = opt.get_int("-nsrc");
    int type = opt.is_set("-sp") ? OSKAR_SINGLE : OSKAR_DOUBLE;
    int jones_type = type | OSKAR_COMPLEX;
    if (!opt.is_set("-s"))
    {
        jones_type |= OSKAR_MATRIX;
    }
    int niter = opt.get_int("-n");
    int use_extended = opt.is_set("-e") ? OSKAR_TRUE : OSKAR_FALSE;
    int use_bandwidth_smearing = opt.is_set("-b") ? OSKAR_TRUE : OSKAR_FALSE;
    int use_time_smearing = opt.is_set("-t") ? OSKAR_TRUE : OSKAR_FALSE;
    std::string raw_file, ascii_file;
    if (opt.is_set("-r"))
    {
        raw_file = opt.get_string("-r");
    }
    if (opt.is_set("-a"))
    {
        ascii_file = opt.get_string("-a");
    }
    if (opt.is_set("-std"))
    {
        max_std_dev = opt.get_double("-std");
    }
    if (opt.is_set("-g"))
    {
        location = OSKAR_GPU;
    }
    if (opt.is_set("-c"))
    {
        location = OSKAR_CPU;
    }
    if (opt.is_set("-cl"))
    {
        location = OSKAR_CL;
    }
    if (location < 0)
    {
        opt.error("Please select one of -g, -c or -cl");
        return EXIT_FAILURE;
    }

    if (opt.is_set("-v"))
    {
        printf("\n");
        printf("- Number of stations: %i\n", num_stations);
        printf("- Number of sources: %i\n", num_sources);
        printf("- Precision: %s\n", (type == OSKAR_SINGLE) ? "single" : "double");
        printf("- Jones type: %s\n", (opt.is_set("-s")) ? "scalar" : "matrix");
        printf("- Extended sources: %s\n", (use_extended) ? "true" : "false");
        printf("- Bandwidth smearing: %s\n", (use_bandwidth_smearing) ?
                "true" : "false");
        printf("- Time smearing: %s\n", (use_time_smearing) ?
                "true" : "false");
        printf("- Number of iterations: %i\n", niter);
        if (max_std_dev > 0.0)
        {
            printf("- Max standard deviations: %f\n", max_std_dev);
        }
        if (!raw_file.empty())
        {
            printf("- Writing iteration data to: %s\n", raw_file.c_str());
        }
        printf("\n");
    }

    // Run benchmarks.
    oskar_device_set_require_double_precision(type == OSKAR_DOUBLE);
    double time_taken_sec = 0.0, average_time_sec = 0.0;
    std::vector<double> times;
    benchmark(num_stations, num_sources, type, jones_type, location,
            use_extended, use_bandwidth_smearing, use_time_smearing,
            niter, times, ascii_file, &status);

    // Compute total time taken.
    for (int i = 0; i < niter; ++i)
    {
        time_taken_sec += times[i];
    }

    // Dump raw data if requested.
    if (!raw_file.empty())
    {
        FILE* raw_stream = 0;
        raw_stream = fopen(raw_file.c_str(), "w");
        if (raw_stream)
        {
            for (int i = 0; i < niter; ++i)
            {
                fprintf(raw_stream, "%.6f\n", times[i]);
            }
            fclose(raw_stream);
        }
    }

    // Check for errors.
    if (status)
    {
        fprintf(stderr, "ERROR: correlate failed with code %i: %s\n", status,
                oskar_get_error_string(status));
        return EXIT_FAILURE;
    }

    // Compute average.
    if (max_std_dev > 0.0)
    {
        double std_dev_sec = 0.0, old_time_average_sec = 0.0;

        // Compute standard deviation.
        old_time_average_sec = time_taken_sec / niter;
        for (int i = 0; i < niter; ++i)
        {
            std_dev_sec += pow(times[i] - old_time_average_sec, 2.0);
        }
        std_dev_sec /= niter;
        std_dev_sec = sqrt(std_dev_sec);

        // Compute new mean.
        average_time_sec = 0.0;
        int counter = 0;
        for (int i = 0; i < niter; ++i)
        {
            if (fabs(times[i] - old_time_average_sec) <
                    max_std_dev * std_dev_sec)
            {
                average_time_sec += times[i];
                counter++;
            }
        }
        if (counter)
        {
            average_time_sec /= counter;
        }
    }
    else
    {
        average_time_sec = time_taken_sec / niter;
    }

    // Print average.
    if (opt.is_set("-v"))
    {
        printf("==> Total time taken: %f seconds.\n", time_taken_sec);
        printf("==> Time taken per iteration: %f seconds.\n", average_time_sec);
        printf("==> Iteration values:\n");
        for (int i = 0; i < niter; ++i)
        {
            printf("%.6f\n", times[i]);
        }
        printf("\n");
    }
    else
    {
        printf("%f\n", average_time_sec);
    }

    return EXIT_SUCCESS;
}


void benchmark(int num_stations, int num_sources, int type,
        int jones_type, int location, int use_extended,
        int use_bandwidth_smearing, int use_time_smearing,
        int niter, std::vector<double>& times, const std::string& ascii_file,
        int* status)
{
    oskar_Timer* timer = oskar_timer_create(location);

    // Create arrays and fill inputs with random data in sensible ranges.
    srand(2);
    oskar_Mem *src_dir[3], *src_ext[3], *src_flux[4], *uvw[3];
    oskar_Telescope* tel = oskar_telescope_create(
            type, location, num_stations, status);
    oskar_Jones* J = oskar_jones_create(
            jones_type, location, num_stations, num_sources, status);
    oskar_Mem* vis = oskar_mem_create(
            jones_type, location, oskar_telescope_num_baselines(tel), status);
    for (int i = 0; i < 4; ++i)
    {
        src_flux[i] = oskar_mem_create(type, location, num_sources, status);
    }
    for (int i = 0; i < 3; ++i)
    {
        src_dir[i] = oskar_mem_create(type, location, num_sources, status);
        src_ext[i] = oskar_mem_create(type, location, num_sources, status);
        uvw[i] = oskar_mem_create(type, location, num_stations, status);
        oskar_mem_random_range(uvw[i], 1.0, 5.0, status);
        oskar_mem_random_range(src_dir[i], 0.1, 0.9, status);
        oskar_mem_random_range(src_ext[i], 0.1e-6, 0.2e-6, status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, i),
                0.1, 1000.0, status);
    }
    oskar_mem_random_range(oskar_jones_mem(J), 1.0, 5.0, status);
    oskar_mem_random_range(src_flux[0], 1.0, 2.0, status);
    oskar_mem_random_range(src_flux[1], 0.1, 1.0, status);
    oskar_mem_random_range(src_flux[2], 0.1, 0.5, status);
    oskar_mem_random_range(src_flux[3], 0.1, 0.2, status);

    // Set options for bandwidth smearing, time smearing, extended sources.
    oskar_telescope_set_channel_bandwidth(tel, 10e6 * use_bandwidth_smearing);
    oskar_telescope_set_time_average(tel, 10 * use_time_smearing);

    // Run benchmark.
    times.resize(niter);
    char* device_name = oskar_device_name(location, 0);
    printf("Using device '%s'\n", device_name);
    free(device_name);
    for (int i = 0; i < niter; ++i)
    {
        oskar_mem_clear_contents(vis, status);
        oskar_timer_start(timer);
        oskar_cross_correlate(use_extended, num_sources, J,
                src_flux, src_dir, src_ext,
                tel, uvw, 0.0, 100e6, 0, vis, status);
        times[i] = oskar_timer_elapsed(timer);
    }

    // Save visibility data if required.
    if (!*status && !ascii_file.empty())
    {
        FILE* fhan = fopen(ascii_file.c_str(), "w");
        if (fhan)
        {
            oskar_mem_save_ascii(fhan, 1, 0, oskar_mem_length(vis),
                    status, vis);
            fclose(fhan);
        }
    }

    // Free memory.
    oskar_mem_free(vis, status);
    oskar_jones_free(J, status);
    oskar_telescope_free(tel, status);
    for (int i = 0; i < 3; ++i)
    {
        oskar_mem_free(src_dir[i], status);
        oskar_mem_free(src_ext[i], status);
        oskar_mem_free(uvw[i], status);
    }
    for (int i = 0; i < 4; ++i)
    {
        oskar_mem_free(src_flux[i], status);
    }
    oskar_timer_free(timer);
}
