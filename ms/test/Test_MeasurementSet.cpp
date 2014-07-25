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

#include <gtest/gtest.h>

#include "ms/oskar_MeasurementSet.h"
#include <vector>
#include <complex>

TEST(MeasurementSet, test_create_simple)
{
    // Create the Measurement Set.
    oskar_MeasurementSet ms;

    // Add some dummy antenna positions.
    double ax[] = {0, 0, 0};
    double ay[] = {0, 0, 0};
    double az[] = {0, 0, 0};
    int na = sizeof(ax) / sizeof(double);
    ms.create("simple.ms", 1, 1, na);
    ms.addAntennas(na, ax, ay, az);

    // Add the Right Ascension & Declination of field centre.
    ms.addField(0, 1.570796);

    // Add frequency band.
    ms.addBand(0, 1, 400e6, 1.0);

    // Add test visibilities (don't include conjugated versions).
    double u[] = {1000.0, 2000.01, 156.03};
    double v[] = {0.0, -241.02, 1678.04};
    double w[] = {0.0, -56.0, 145.0};
    double vis[] = {1.0, 0.0, 0.00, 0.0, 0.00, 0.0};
    double times[] = {1.0, 2.0, 3.0};
    int ant1[] = {0, 0, 1};
    int ant2[] = {1, 2, 2};
    int nv = sizeof(u) / sizeof(double);
    ms.addVisibilities(1, 1, nv, u, v, w, vis, ant1, ant2, 90, 90, times);
}

TEST(MeasurementSet, test_multi_channel)
{
    // Define the data dimensions.
    int n_ant = 3;           // Number of antennas.
    int n_pol = 4;           // Number of polarisations.
    int n_chan = 10;         // Number of channels.
    int n_dumps = 2;         // Number of total correlator dumps.

    // Define other meta-data.
    const char* filename = "multi_channel.ms";
    double ra = 0.0;          // RA of field centre in radians.
    double dec = 1.570796;    // Dec of field centre in radians.
    double exposure = 90.0;   // Visibility exposure time in seconds.
    double interval = 90.0;   // Visibility dump interval in seconds.
    double freq = 400e6;      // Frequency of channel 0 in Hz.
    double chan_width = 25e3; // Channel width in Hz.

    // Create the Measurement Set.
    oskar_MeasurementSet ms;
    ms.create(filename, n_pol, n_chan, n_ant);

    // Add some dummy antenna positions.
    std::vector<double> ax(n_ant), ay(n_ant), az(n_ant);
    for (int i = 0; i < n_ant; ++i)
    {
        ax[i] = i / 10.0;
        ay[i] = i / 20.0;
        az[i] = i / 30.0;
    }
    ms.addAntennas(n_ant, &ax[0], &ay[0], &az[0]);

    // Add the Right Ascension & Declination of field centre.
    ms.addField(ra, dec, "Test field");

    // Add polarisations.
    ms.addPolarisation(n_pol);

    // Add frequency band.
    ms.addBand(0, n_chan, freq, chan_width);

    // Create test data (without complex conjugate).
    int n_baselines = n_ant * (n_ant - 1) / 2;
    int n_rows = n_baselines * n_dumps;
    std::vector<double> u(n_rows), v(n_rows), w(n_rows), times(n_rows);
    std::vector< std::complex<double> > vis_data(n_pol * n_chan * n_rows);
    std::vector<int> ant1(n_rows), ant2(n_rows);

    // Fill the vectors.
    for (int d = 0, r = 0; d < n_dumps; ++d)
    {
        for (int ai = 0; ai < n_ant; ++ai)
        {
            for (int aj = ai+1; aj < n_ant; ++aj)
            {
                // Create the u,v,w coordinates.
                u[r] = 10.0 * r;
                v[r] = 50.0 * r;
                w[r] = 100.0 * r;

                // Create the visibility data.
                for (int c = 0; c < n_chan; ++c)
                {
                    for (int p = 0; p < n_pol; ++p)
                    {
                        int vi = r * n_pol * n_chan + c * n_pol + p;
                        double re = (p + 1) * (c + 1) * 10.0;
                        double im = (double)r;
                        vis_data[vi] = std::complex<double>(re, im);
                    }
                }

                // Create the antenna index pairs.
                ant1[r] = ai;
                ant2[r] = aj;

                // Create the time stamps.
                times[r] = (double)r;

                // Increment the row index.
                ++r;
            }
        }
    }

    ms.addVisibilities(n_pol, n_chan, n_rows, &u[0], &v[0], &w[0],
            (double*)(&vis_data[0]), &ant1[0], &ant2[0], exposure, interval,
            &times[0]);
}
