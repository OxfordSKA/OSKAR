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

#include "station/test/Test_evaluate_jones_E.h"

#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_sky_model_save.h"
#include "math/oskar_Jones.h"
#include "math/oskar_jones_get_station_pointer.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "math/oskar_SphericalPositions.h"
#include "station/oskar_evaluate_jones_E.h"
#include "station/oskar_station_model_copy.h"
#include "station/oskar_station_model_multiply_by_wavenumber.h"
#include "utility/oskar_curand_state_free.h"
#include "utility/oskar_curand_state_init.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_vector_types.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#ifndef c_0
#define c_0 299792458.0
#endif

void Test_evaluate_jones_E::test_fail_conditions()
{
}

void Test_evaluate_jones_E::evaluate_e()
{
    float deg2rad = M_PI / 180.0;

    int error = 0;

    int gast = 0.0;

    // Construct telescope model.
    // - construct a station model and load it into the telescope structure.
    double frequency = 30e6;
    int station_dim = 100;
    double station_size_m = 180.0;
    int num_antennas = station_dim * station_dim;
    oskar_StationModel station_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_antennas);
    station_cpu.longitude_rad = 0.0;
    station_cpu.latitude_rad  = M_PI_2;
    station_cpu.altitude_m  = 0.0;
    station_cpu.coord_units = OSKAR_METRES;
    float* x_pos = (float*) malloc(station_dim * sizeof(float));
    oskar_linspace_f(x_pos, -station_size_m/2.0, station_size_m/2.0, station_dim);
    oskar_meshgrid_f(station_cpu.x_weights, station_cpu.y_weights, x_pos, station_dim,
            x_pos, station_dim);
    free(x_pos);
    station_cpu.ra0_rad  = 0.0;
    station_cpu.dec0_rad = M_PI_2;

    // Set the station meta-data.

    int num_stations = 2;
    oskar_TelescopeModel telescope_gpu(OSKAR_SINGLE, OSKAR_LOCATION_GPU, num_stations);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_station_model_copy(&telescope_gpu.station[i], &station_cpu, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        oskar_station_model_multiply_by_wavenumber(&(telescope_gpu.station[i]),
                frequency, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        telescope_gpu.station[i].apply_element_errors = OSKAR_FALSE;
        telescope_gpu.station[i].element_pattern->type = OSKAR_ELEMENT_MODEL_TYPE_ISOTROPIC;
        telescope_gpu.station[i].use_polarised_elements = false;
    }
    telescope_gpu.identical_stations = OSKAR_TRUE;
    telescope_gpu.use_common_sky = OSKAR_TRUE;
    // Other fields of the telescope structure are not used for E?!

    // Initialise the random number generator.
    oskar_CurandState curand_state;
    int seed = 0;
    oskar_curand_state_init(&curand_state, num_antennas, seed, 0, 0, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Construct a sky model.
    int num_sources = 0;
    oskar_SkyModel sky_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_sources);
    float centre_long = 180.0 * deg2rad;
    float centre_lat  = 0.0   * deg2rad;
    float size_long   = 180.0 * deg2rad;
    float size_lat    = 90.0  * deg2rad;
    float sep_long    = 1.0   * deg2rad;
    float sep_lat     = 1.0   * deg2rad;
    float rho         = 0.0   * deg2rad;
    bool force_constant_sep = true;
    bool set_centre_after   = false;
    bool force_centre_point = true;
    bool force_to_edges     = true;
    int projection_type = oskar_SphericalPositions<float>::PROJECTION_NONE;
    oskar_SphericalPositions<float> positions(centre_long, centre_lat,
            size_long, size_lat, sep_long, sep_lat, force_constant_sep,
            set_centre_after, rho, force_centre_point, force_to_edges,
            projection_type);
    num_sources = positions.generate(0, 0);
    oskar_sky_model_resize(&sky_cpu, num_sources, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    positions.generate(sky_cpu.RA, sky_cpu.Dec);

//    error = oskar_sky_model_write("temp_test_sky.txt", &sky_cpu);
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy sky to the GPU
    oskar_SkyModel sky_gpu(&sky_cpu, OSKAR_LOCATION_GPU);

    oskar_Jones E_gpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, num_stations,
            num_sources);

    oskar_WorkStationBeam work_gpu(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
    oskar_evaluate_jones_E(&E_gpu, &sky_gpu, &telescope_gpu, gast,
            &work_gpu, &curand_state, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy the Jones matrix back to the CPU.
    oskar_Jones E_cpu(&E_gpu, OSKAR_LOCATION_CPU);

    // Save to file for plotting.
    oskar_Mem l(&work_gpu.hor_x, OSKAR_LOCATION_CPU);
    oskar_Mem m(&work_gpu.hor_y, OSKAR_LOCATION_CPU);
    oskar_Mem n(&work_gpu.hor_z, OSKAR_LOCATION_CPU);
    const char* filename = "temp_test_E_jones.txt";
    FILE* file = fopen(filename, "w");
    oskar_Mem E_station;
    for (int j = 0; j < num_stations; ++j)
    {
        oskar_jones_get_station_pointer(&E_station, &E_cpu, j, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
        for (int i = 0; i < num_sources; ++i)
        {
            fprintf(file, "%10.3f,%10.3f,%10.3f,%10.3f,%10.3f\n",
                    ((float*)l.data)[i],
                    ((float*)m.data)[i],
                    ((float*)n.data)[i],
                    ((float2*)(E_station.data))[i].x,
                    ((float2*)(E_station.data))[i].y);
        }
    }
    fclose(file);

    /*
        data = dlmread('temp_test_E_jones.txt');
        l  = reshape(data(:,1), length(data(:,1))/2, 2);
        m  = reshape(data(:,2), length(data(:,2))/2, 2);
        n  = reshape(data(:,3), length(data(:,3))/2, 2);
        re = reshape(data(:,4), length(data(:,4))/2, 2);
        im = reshape(data(:,5), length(data(:,5))/2, 2);
        amp = sqrt(re.^2 + im.^2);
        %idx = find(n > 0.0);
        %l = l(idx);
        %m = m(idx);
        %n = n(idx);
        %amp = amp(idx);
        station = 1;
        scatter3(l(:,station),m(:,station),n(:,station),2,amp(:,station));
     */
    oskar_curand_state_free(&curand_state, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
}


void Test_evaluate_jones_E::performance_test()
{
}
