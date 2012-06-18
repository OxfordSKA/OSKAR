/*
 * Copyright (c) 2012, The University of Oxford
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


#include "apps/lib/test/Test_write_ms.h"
#include "apps/lib/oskar_visibilities_write_ms.h"

#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_SettingsTime.h"

#include "sky/oskar_date_time_to_mjd.h"

#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"

#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void Test_write_ms::test_write()
{
    int num_antennas  = 5;
    int num_channels  = 3;
    int num_times     = 5;
    int num_baselines = num_antennas * (num_antennas - 1) / 2;

    // Create a visibility structure and fill in some data.
    oskar_Visibilities vis(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU,
            num_channels, num_times, num_antennas);
    vis.phase_centre_ra_deg = 160.0;
    vis.phase_centre_dec_deg = 89.0;
    for (int i = 0, c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
        {
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                // XX
                ((double4c*)vis.amplitude.data)[i].a.x = (double)c + 0.1;
                ((double4c*)vis.amplitude.data)[i].a.y = 0.05;
                // XY
                ((double4c*)vis.amplitude.data)[i].b.x = (double)t + 0.1;
                ((double4c*)vis.amplitude.data)[i].b.y = 0.15;
                // YX
                ((double4c*)vis.amplitude.data)[i].c.x = (double)b + 0.1;
                ((double4c*)vis.amplitude.data)[i].c.y = 0.25;
                // YY
                ((double4c*)vis.amplitude.data)[i].d.x = (double)i + 0.1;
                ((double4c*)vis.amplitude.data)[i].d.y = 0.35;
            }
        }
    }
    for (int i = 0, t = 0; t < num_times; ++t)
    {
        for (int b = 0; b < num_baselines; ++b, ++i)
        {
            ((double*)vis.uu_metres)[i] = (double)t + 0.001;
            ((double*)vis.vv_metres)[i] = (double)b + 0.002;
            ((double*)vis.ww_metres)[i] = (double)i + 0.003;
        }
    }
    for (int i = 0; i < num_antennas; ++i)
    {
        ((double*)vis.x_metres)[i] = (double)i + 0.1;
        ((double*)vis.y_metres)[i] = (double)i + 0.2;
        ((double*)vis.z_metres)[i] = (double)i + 0.3;
    }
    vis.freq_start_hz      = 222.22e6;
    vis.freq_inc_hz        = 11.1e6;
    vis.time_start_mjd_utc = oskar_date_time_to_mjd(2011, 11, 17, 0.0);
    vis.time_inc_seconds   = 1.0;

    const char* filename = "temp_test_write_ms.ms";

    int error = oskar_visibilities_write_ms(&vis, NULL, filename, OSKAR_TRUE);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
}

