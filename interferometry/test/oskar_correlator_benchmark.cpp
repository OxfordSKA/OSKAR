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
#include <interferometry/oskar_TelescopeModel.h>
#include <interferometry/oskar_telescope_model_init.h>
#include <sky/oskar_SkyModel.h>
#include <sky/oskar_sky_model_init.h>
#include <math/oskar_Jones.h>
#include <math/oskar_jones_init.h>
#include <utility/oskar_Mem.h>
#include <utility/oskar_mem_init.h>
#include <utility/oskar_mem_copy.h>
#include <utility/oskar_get_error_string.h>
#define TIMER_ENABLE
#include <utility/timer.h>

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstdio>

int benchmark(int num_stations, int num_sources, int type,
        int jones_type, int use_extended, int use_time_ave, int niter,
        double* time_taken_sec);

int main(int /*argc*/, char** /*argv*/)
{
    int num_stations = 100;
    int num_sources = 10000;
    int type = OSKAR_DOUBLE;
    int jones_type = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    int use_extended = OSKAR_FALSE;
    int use_time_ave = OSKAR_TRUE;
    int niter = 1;
    double time_taken_sec = 0.0;

    int status = benchmark(num_stations, num_sources, type, jones_type,
            use_extended, use_time_ave, niter, &time_taken_sec);
    if (status) {
        printf("ERROR: correlator failed with code %i: %s\n", status,
                oskar_get_error_string(status));
    }

    return EXIT_SUCCESS;
}

int benchmark(int num_stations, int num_sources, int type,
        int jones_type, int use_extended, int use_time_ave, int niter,
        double* time_taken_sec)
{
    int status = OSKAR_SUCCESS;
    int loc = OSKAR_LOCATION_GPU;
    int num_vis = num_stations * (num_stations-1) / 2;
    int num_vis_coords = num_stations;

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
    TIMER_START
    {
        for (int i = 0; i < niter; ++i)
            oskar_correlate(&vis, &J, &tel, &sky, &u, &v, gast, &status);
    }
    cudaDeviceSynchronize();
    TIMER_STOP("oskar_correlate");

    *time_taken_sec = 0.0;

    // Note: telescope, sky model, oskar_Mem, and oskar_Jones currently still
    // have a C++ nature so free themselves!

    return status;
}
