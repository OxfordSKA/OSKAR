/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_VisData.h"
#include "apps/lib/oskar_Settings.h"
#include "apps/lib/oskar_write_ms.h"
#include "apps/lib/oskar_file_utils.h"

#include <QtCore/QFile>
#include <QtCore/QString>
#include <QtCore/QTextStream>
#include <QtCore/QStringList>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

int main(int /*argc*/, char** /*argv*/)
{
    // Create a telescope data file with one antenna.
    QString telescope_file = "temp_telescope.dat";
    QFile file(telescope_file);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return EXIT_FAILURE;
    QTextStream out(&file);
    int num_stations = 10;
    vector<double> x_horizon(num_stations);
    vector<double> y_horizon(num_stations);
    for (int i = 0; i < num_stations; ++i)
    {
        x_horizon[i] = (float)i + i/10.0;
        y_horizon[i] = (float)i - i/10.0;
        out << x_horizon[i] << "," << y_horizon[i] << endl;
    }
    file.close();

    // Create some visibility data to write.
    oskar_VisData_d vis;
    int num_baselines = num_stations * (num_stations - 1) / 2;
    int num_dumps     = 2;
    int num_samples   = num_baselines * num_dumps;
    oskar_allocate_vis_data_d(num_samples, &vis);
    for (int i = 0; i < num_samples; ++i)
    {
        vis.u[i]     = (double)i;
        vis.v[i]     = (double)i + 0.1;
        vis.w[i]     = (double)i + 0.2;
        vis.amp[i].x = 1.0 + (double)i / 1000.0;
        vis.amp[i].y = 0.0;
    }


    // Construct the required settings.
    oskar_Settings settings;
    settings.obs().set_ra0_deg(0.0);
    settings.obs().set_dec0_deg(90.0);
    settings.obs().set_start_frequency(250e6);
    settings.obs().set_num_channels(1);
    settings.obs().set_frequency_inc(0.0);
    settings.obs().set_start_time_utc_mjd(0.0);
    settings.obs().set_obs_length_sec(60.0 * 2.0);
    settings.obs().set_num_vis_dumps(num_dumps);
    settings.set_telescope_file(telescope_file);
    settings.set_longitude_deg(0.0);
    settings.set_latitude_deg(90.0);

    // Write the MS.
    const char* ms_path = "temp_test.ms";
    oskar_write_ms_d(ms_path, &settings, &vis, 0);

    // Free memory.
    oskar_free_vis_data_d(&vis);

    // Cleanup temporary files.
    QFile::remove(telescope_file);

    return EXIT_SUCCESS;
}

