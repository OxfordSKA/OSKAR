/*
 * Copyright (c) 2013, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <oskar_global.h>

#include <interferometry/oskar_correlate.h>
#include <interferometry/oskar_telescope_model_free.h>
#include <interferometry/oskar_telescope_model_init.h>
#include <sky/oskar_sky_model_free.h>
#include <sky/oskar_sky_model_init.h>
#include <math/oskar_jones_free.h>
#include <math/oskar_jones_init.h>
#include <utility/oskar_mem_copy.h>
#include <utility/oskar_mem_free.h>
#include <utility/oskar_mem_init.h>
#include <utility/oskar_get_error_string.h>

#include <apps/lib/oskar_OptionParser.h>
#include <cuda_runtime_api.h>

#ifndef _WIN32
#   include <sys/time.h>
#endif /* _WIN32 */
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

int benchmark(int num_stations, int num_sources, int type,
        int jones_type, int use_extended, int use_time_ave, int niter,
        std::vector<double>& times);

int main(int argc, char** argv)
{
    oskar_OptionParser opt("oskar_correlator_benchmark");
    opt.addFlag("-nst", "Number of stations.", 1, "", true);
    opt.addFlag("-nsrc", "Number of sources.", 1, "", true);
    opt.addFlag("-sp", "Use single precision (default = double precision)");
    opt.addFlag("-s", "Use scalar Jones terms (default = matrix/polarised).");
    opt.addFlag("-e", "Use extended (Gaussian) sources (default = point sources).");
    opt.addFlag("-t", "Use analytical time averaging (default = no time "
            "averaging).");
    opt.addFlag("-r", "Dump raw iteration data to this file.", 1);
    opt.addFlag("-std", "Discard values greater than this number of standard "
            "deviations from the mean.", 1);
    opt.addFlag("-n", "Number of iterations", 1, "1", false);
    opt.addFlag("-v", "Display verbose output.", false);
    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    int num_stations, num_sources, niter;
    double max_std_dev = 0.0;
    opt.get("-nst")->getInt(num_stations);
    opt.get("-nsrc")->getInt(num_sources);
    int type = opt.isSet("-sp") ? OSKAR_SINGLE : OSKAR_DOUBLE;
    int jones_type = type | OSKAR_COMPLEX;
    if (!opt.isSet("-s"))
        jones_type |= OSKAR_MATRIX;
    opt.get("-n")->getInt(niter);
    int use_extended = opt.isSet("-e") ? OSKAR_TRUE : OSKAR_FALSE;
    int use_time_ave = opt.isSet("-t") ? OSKAR_TRUE : OSKAR_FALSE;
    std::string raw_file;
    if (opt.isSet("-r"))
        opt.get("-r")->getString(raw_file);
    if (opt.isSet("-std"))
        opt.get("-std")->getDouble(max_std_dev);

    if (opt.isSet("-v"))
    {
        printf("\n");
        printf("- Number of stations: %i\n", num_stations);
        printf("- Number of sources: %i\n", num_sources);
        printf("- Precision: %s\n", (type == OSKAR_SINGLE) ? "single" : "double");
        printf("- Jones type: %s\n", (opt.isSet("-s")) ? "Scalar" : "Matrix");
        printf("- Extended sources: %s\n", (use_extended) ? "true" : "false");
        printf("- Analytical time smearing: %s\n", (use_time_ave) ? "true" : "false");
        printf("- Number of iterations: %i\n", niter);
        if (max_std_dev > 0.0)
            printf("- Max standard deviations: %f\n", max_std_dev);
        if (!raw_file.empty())
            printf("- Writing iteration data to: %s\n", raw_file.c_str());
        printf("\n");
    }

    // Run benchmarks.
    double time_taken_sec = 0.0, average_time_sec = 0.0;
    std::vector<double> times;
    int status = benchmark(num_stations, num_sources, type, jones_type,
            use_extended, use_time_ave, niter, times);

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
        fprintf(stderr, "ERROR: correlator failed with code %i: %s\n", status,
                oskar_get_error_string(status));
        return EXIT_FAILURE;
    }

    // Compute average.
    if (max_std_dev > 0.0)
    {
        double std_dev_sec = 0.0, old_time_average_sec;

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
            average_time_sec /= counter;
    }
    else
    {
        average_time_sec = time_taken_sec / niter;
    }

    // Print average.
    if (opt.isSet("-v"))
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

int benchmark(int num_stations, int num_sources, int type,
        int jones_type, int use_extended, int use_time_ave, int niter,
        std::vector<double>& times)
{
    int status = OSKAR_SUCCESS;
    int loc = OSKAR_LOCATION_GPU;
    int num_vis = num_stations * (num_stations-1) / 2;
    int num_vis_coords = num_stations;

    // Set device 0.
    cudaSetDevice(0);

    // Create the CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double time_ave = 0.0;
    if (use_time_ave)
        time_ave = 1.0;

    // Setup a test telescope model.
    oskar_TelescopeModel tel;
    oskar_telescope_model_init(&tel, type, loc, num_stations, &status);
    tel.time_average_sec = time_ave;

    oskar_SkyModel sky;
    oskar_sky_model_init(&sky, type, loc, num_sources, &status);
    sky.use_extended = use_extended;

    // Memory for the visibility slice being correlated.
    oskar_Mem vis;
    oskar_mem_init(&vis, jones_type, loc, num_vis, OSKAR_TRUE, &status);

    // Visibility coordinates.
    oskar_Mem u, v;
    oskar_mem_init(&u, type, loc, num_vis_coords, OSKAR_TRUE, &status);
    oskar_mem_init(&v, type, loc, num_vis_coords, OSKAR_TRUE, &status);

    oskar_Jones J;
    oskar_jones_init(&J, jones_type,
            loc, num_stations, num_sources, &status);
    if (status) return status;

    double gast = 0.0;
    cudaDeviceSynchronize();

    times.resize(niter);
    for (int i = 0; i < niter; ++i)
    {
        float millisec = 0.0f;
        cudaEventRecord(start);
        oskar_correlate(&vis, &J, &tel, &sky, &u, &v, gast, &status);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millisec, start, stop);

        // Store the time taken for this iteration.
        times[i] = millisec / 1000.0;
    }

    // Destroy the timers.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory.
    oskar_mem_free(&u, &status);
    oskar_mem_free(&v, &status);
    oskar_mem_free(&vis, &status);
    oskar_jones_free(&J, &status);
    oskar_sky_model_free(&sky, &status);
    oskar_telescope_model_free(&tel, &status);
    return status;
}
