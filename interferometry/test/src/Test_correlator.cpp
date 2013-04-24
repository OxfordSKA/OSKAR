/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <cuda_runtime_api.h>

#include "interferometry/test/Test_correlator.h"
#include "interferometry/oskar_correlate.h"
#include "interferometry/oskar_telescope_model_free.h"
#include "interferometry/oskar_telescope_model_init.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_init.h"
#include "math/oskar_jones_free.h"
#include "math/oskar_jones_init.h"
#include "utility/oskar_cuda_device_info_scan.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#define TIMER_ENABLE
#include "utility/timer.h"

Test_correlator::Test_correlator()
{
    device_ = new oskar_CudaDeviceInfo;
    oskar_cuda_device_info_scan(device_, 0);
}

Test_correlator::~Test_correlator()
{
    delete device_;
}

void Test_correlator::benchmark()
{
    int num_stations = 100;
    int num_sources = 10000;
    int niter = 1;
    int type, jones_type, use_extended, use_time_ave;

    // Single precision.
    type = OSKAR_SINGLE;
    jones_type = type | OSKAR_COMPLEX | OSKAR_MATRIX;

    // Point sources, no time smearing.
    use_extended = OSKAR_FALSE;
    use_time_ave = OSKAR_FALSE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
            use_time_ave, niter, "correlator (single, point, no time avg)");

    // Point sources, with time smearing.
    use_extended = OSKAR_FALSE;
    use_time_ave = OSKAR_TRUE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (single, point, time avg)");

    // Extended sources, no time smearing.
    use_extended = OSKAR_TRUE;
    use_time_ave = OSKAR_FALSE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (single, extended, no time avg)");

    // Extended sources, with time smearing.
    use_extended = OSKAR_TRUE;
    use_time_ave = OSKAR_TRUE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (single, extended, time avg)");

    // Return if device has no double precision support.
    if (!device_->supports_double)
        return;

    // Double precision.
    type = OSKAR_DOUBLE;
    jones_type = type | OSKAR_COMPLEX | OSKAR_MATRIX;

    // Point sources, no time smearing.
    use_extended = OSKAR_FALSE;
    use_time_ave = OSKAR_FALSE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
            use_time_ave, niter, "correlator (double, point, no time avg)");

    // Point sources, with time smearing.
    use_extended = OSKAR_FALSE;
    use_time_ave = OSKAR_TRUE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (double, point, time avg)");

    // Extended sources, no time smearing.
    use_extended = OSKAR_TRUE;
    use_time_ave = OSKAR_FALSE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (double, extended, no time avg)");

    // Extended sources, with time smearing.
    use_extended = OSKAR_TRUE;
    use_time_ave = OSKAR_TRUE;
    benchmark_(num_stations, num_sources, jones_type, use_extended,
                use_time_ave, niter, "correlator (double, extended, time avg)");
}

void Test_correlator::benchmark_(int num_stations, int num_sources,
        int jones_type, int use_extended, int use_time_ave, int niter,
        const char* message)
{
    int status = OSKAR_SUCCESS;
    int loc = OSKAR_LOCATION_GPU;
    int num_vis = num_stations * (num_stations-1) / 2;
    int num_vis_coords = num_stations;
    int type;

    // Get the base data type.
    type = oskar_mem_base_type(jones_type);

    double time_ave = 0.0;
    if (use_time_ave)
        time_ave = 1.0;

    // Set up a test telescope model.
    oskar_TelescopeModel tel;
    oskar_telescope_model_init(&tel, type, loc, num_stations, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);
    tel.time_average_sec = time_ave;

    oskar_SkyModel sky;
    oskar_sky_model_init(&sky, type, loc, num_sources, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);
    sky.use_extended = use_extended;

    // Memory for the visibility slice being correlated.
    oskar_Mem vis;
    oskar_mem_init(&vis, jones_type, loc, num_vis, OSKAR_TRUE, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);

    // Visibility coordinates.
    oskar_Mem u, v;
    oskar_mem_init(&u, type, loc, num_vis_coords, OSKAR_TRUE, &status);
    oskar_mem_init(&v, type, loc, num_vis_coords, OSKAR_TRUE, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);

    oskar_Jones J;
    oskar_jones_init(&J, jones_type,
            loc, num_stations, num_sources, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);

    double gast = 0.0;
    TIMER_START
    {
        for (int i = 0; i < niter; ++i)
            oskar_correlate(&vis, &J, &tel, &sky, &u, &v, gast, &status);
    }
    cudaDeviceSynchronize();
    TIMER_STOP("%s", message);

    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);

    // Free memory.
    oskar_jones_free(&J, &status);
    oskar_mem_free(&u, &status);
    oskar_mem_free(&v, &status);
    oskar_mem_free(&vis, &status);
    oskar_telescope_model_free(&tel, &status);
    oskar_sky_model_free(&sky, &status);
}
