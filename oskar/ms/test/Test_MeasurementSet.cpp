/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include <gtest/gtest.h>
#include "ms/oskar_measurement_set.h"
#include <vector>
#include <complex>

TEST(MeasurementSet, test_create_simple)
{
    oskar_MeasurementSet* ms;

    // Add some dummy antenna positions.
    double ax[] = {0, 0, 0};
    double ay[] = {0, 0, 0};
    double az[] = {0, 0, 0};
    int na = sizeof(ax) / sizeof(double);
    ms = oskar_ms_create("simple.ms", "test", na, 1, 1, 400e6, 1.0, 0, 1);
    ASSERT_TRUE(ms);
    oskar_ms_set_phase_centre(ms, 0, 0.0, 1.570796);
    oskar_ms_set_station_coords_d(ms, na, ax, ay, az);

    // Add test visibilities (don't include conjugated versions).
    double u[] = {1000.0, 2000.01, 156.03};
    double v[] = {0.0, -241.02, 1678.04};
    double w[] = {0.0, -56.0, 145.0};
    double vis[] = {1.0, 0.0, 0.00, 0.0, 0.00, 0.0};
    int num_baselines = sizeof(u) / sizeof(double);
    oskar_ms_write_coords_d(ms, 0, num_baselines, u, v, w, 90.0, 90.0, 1.0);
    oskar_ms_write_vis_d(ms, 0, 0, 1, num_baselines, vis);
    oskar_ms_close(ms);
}


TEST(MeasurementSet, test_multi_channel)
{
    int status = 0;

    // Define the data dimensions.
    int n_ant = 3;           // Number of antennas.
    int n_pol = 4;           // Number of polarisations.
    int n_chan = 10;         // Number of channels.
    int n_times = 2;         // Number of correlator dumps.

    // Define other meta-data.
    const char* filename = "multi_channel.ms";
    double ra = 0.0;          // RA of field centre in radians.
    double dec = 1.570796;    // Dec of field centre in radians.
    double exposure = 90.0;   // Visibility exposure time in seconds.
    double interval = 90.0;   // Visibility dump interval in seconds.
    double freq = 400e6;      // Frequency of channel 0 in Hz.
    double chan_width = 25e3; // Channel width in Hz.

    // Create the Measurement Set.
    oskar_MeasurementSet* ms = oskar_ms_create(filename, "test",
            n_ant, n_chan, n_pol, freq, chan_width, 0, 1);
    ASSERT_TRUE(ms);
    oskar_ms_set_phase_centre(ms, 0, ra, dec);

    // Add some dummy antenna positions.
    std::vector<double> ax(n_ant), ay(n_ant), az(n_ant);
    for (int i = 0; i < n_ant; ++i)
    {
        ax[i] = i / 10.0;
        ay[i] = i / 20.0;
        az[i] = i / 30.0;
    }
    oskar_ms_set_station_coords_d(ms, n_ant, &ax[0], &ay[0], &az[0]);

    // Create test data (without complex conjugate).
    int n_baselines = n_ant * (n_ant - 1) / 2;
    std::vector<double> u(n_baselines), v(n_baselines), w(n_baselines);
    std::vector< std::complex<double> > vis_data(n_pol * n_chan * n_baselines);

    // Fill the vectors.
    for (int t = 0; t < n_times; ++t)
    {
        for (int ai = 0, b = 0; ai < n_ant; ++ai)
        {
            for (int aj = ai+1; aj < n_ant; ++b, ++aj)
            {
                // Create the u,v,w coordinates.
                u[b] = 10.0 * (t + 1) + b;
                v[b] = 100.0 * (t + 1) + b;
                w[b] = 1000.0 * (t + 1) + b;
            }
        }
        oskar_ms_write_coords_d(ms, t * n_baselines, n_baselines,
                &u[0], &v[0], &w[0], exposure, interval, (double)t);

        for (int c = 0; c < n_chan; ++c)
        {
            for (int ai = 0, b = 0; ai < n_ant; ++ai)
            {
                for (int aj = ai+1; aj < n_ant; ++b, ++aj)
                {
                    // Create the visibility data.
                    for (int p = 0; p < n_pol; ++p)
                    {
                        int vi = c * n_baselines * n_pol + b * n_pol + p;
                        double re = (p + 1) * (c + 1) * 10.0;
                        double im = 10.0 * (t + 1) + b;
                        vis_data[vi] = std::complex<double>(re, im);
                    }
                }
            }
        }
        oskar_ms_write_vis_d(ms, t * n_baselines, 0, n_chan, n_baselines,
                (double*)(&vis_data[0]));
    }

    // Read the data back again.
    size_t vis_size = n_baselines * n_times * n_chan * n_pol *
            sizeof(std::complex<float>);
    size_t uvw_size = n_baselines * n_times * sizeof(double) * 3;
    size_t required_vis_size = 0, required_uvw_size = 0;
    void* vis = malloc(vis_size);
    void* uvw = malloc(uvw_size);
    oskar_ms_read_column(ms, "DATA", 0, n_baselines * n_times, vis_size, vis,
            &required_vis_size, &status);
    oskar_ms_read_column(ms, "UVW", 0, n_baselines * n_times, uvw_size, uvw,
            &required_uvw_size, &status);
    ASSERT_EQ(0, status);
    ASSERT_EQ(required_vis_size, vis_size);
    ASSERT_EQ(required_uvw_size, uvw_size);

    // Check the data.
    for (int t = 0, r = 0; t < n_times; ++t)
    {
        for (int ai = 0, b = 0; ai < n_ant; ++ai)
        {
            for (int aj = ai+1; aj < n_ant; ++b, ++aj, ++r)
            {
                // Read the u,v,w coordinates.
                ASSERT_EQ(((double*)uvw)[r*3 + 0], 10.0 * (t + 1) + b);
                ASSERT_EQ(((double*)uvw)[r*3 + 1], 100.0 * (t + 1) + b);
                ASSERT_EQ(((double*)uvw)[r*3 + 2], 1000.0 * (t + 1) + b);

                // Read the visibility data.
                for (int c = 0; c < n_chan; ++c)
                {
                    for (int p = 0; p < n_pol; ++p)
                    {
                        int vi = r * n_pol * n_chan + c * n_pol + p;
                        double re = (p + 1) * (c + 1) * 10.0;
                        double im = 10.0 * (t + 1) + b;
                        ASSERT_EQ(((float*)vis)[2 * vi], re);
                        ASSERT_EQ(((float*)vis)[2 * vi + 1], im);
                    }
                }
            }
        }
    }

    // Free memory.
    free(vis);
    free(uvw);
    oskar_ms_close(ms);
}
