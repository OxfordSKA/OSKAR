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

#include "station/test/Test_evaluate_station_beam.h"

#include "oskar_global.h"
#include "station/oskar_evaluate_station_beam.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_station_model_save_config.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_Work.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "utility/oskar_Device_curand_state.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

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

void Test_evaluate_station_beam::test_fail_conditions()
{
}

void Test_evaluate_station_beam::evaluate_test_pattern()
{
    int error = 0;

    double gast = 0.0;

    // Construct a station model.
    double frequency = 30e6;
    int station_dim = 100;
    double station_size_m = 180.0;
    int num_antennas = station_dim * station_dim;
    oskar_StationModel station_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_antennas);

    // Set the station coordinates.
    station_cpu.longitude_rad = 0.0;
    station_cpu.latitude_rad  = M_PI_2;
    station_cpu.altitude_m  = 0.0;
    station_cpu.coord_units = OSKAR_METRES;
    float* x_pos = (float*) malloc(station_dim * sizeof(float));
    oskar_linspace_f(x_pos, -station_size_m/2.0, station_size_m/2.0, station_dim);
    oskar_meshgrid_f(station_cpu.x_weights, station_cpu.y_weights, x_pos, station_dim,
            x_pos, station_dim);
    free(x_pos);
    station_cpu.num_elements = num_antennas;

    // Set the station beam direction.
    station_cpu.ra0_rad  = 0.0;
    station_cpu.dec0_rad = M_PI_2;

    // Set the station meta-data.
    station_cpu.use_polarised_elements = false;

//    error = oskar_station_model_save_configuration("temp_test_station.txt", &station_cpu);
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy the station structure to the gpu and scale the coordinates to wavenumbers.
    oskar_StationModel station_gpu(&station_cpu, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_METRES, station_gpu.coord_units);
    station_gpu.multiply_by_wavenumber(frequency);

    // Evaluate horizontal l,m,n for beam phase centre.
    double beam_l, beam_m, beam_n;
    error = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m, &beam_n,
            &station_gpu, gast);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Evalute horizontal l,m positions at which to generate the beam pattern.
    int image_size = 401;
    double fov_deg = 30.0;
    int num_pixels = image_size * image_size;

    // Generate horizontal lm coordinates for the beam pattern.
    oskar_Mem l_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem m_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem n_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_pixels);
    float* lm = (float*)malloc(image_size * sizeof(float));
    double lm_max = sin(fov_deg * M_PI / 180.0);
    oskar_linspace_f(lm, -lm_max, lm_max, image_size);
    oskar_meshgrid_f(l_cpu, m_cpu, lm, image_size, lm, image_size);
    free(lm);

    // Copy horizontal lm coordinates to GPU.
    oskar_Mem l_gpu(&l_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem m_gpu(&m_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem n_gpu(&n_cpu, OSKAR_LOCATION_GPU);

    // Initialise the random number generator.
    oskar_Device_curand_state curand_state(num_antennas);
    int seed = 0;
    curand_state.init(seed);
    station_gpu.apply_element_errors = OSKAR_FALSE;

    // Allocate weights work array.
    oskar_WorkStationBeam work(OSKAR_SINGLE, OSKAR_LOCATION_GPU);

    // Declare memory for the beam pattern.
    oskar_Mem beam_pattern(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, num_pixels);

    station_gpu.array_is_3d = 0;
    TIMER_START
    error = oskar_evaluate_station_beam(&beam_pattern, &station_gpu, beam_l,
            beam_m, beam_n, num_pixels, &l_gpu, &m_gpu, &n_gpu, &work,
            &curand_state);
    TIMER_STOP("Finished station beam (2D)");
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    station_gpu.array_is_3d = 1;
    TIMER_START
    error = oskar_evaluate_station_beam(&beam_pattern, &station_gpu, beam_l,
            beam_m, beam_n, num_pixels, &l_gpu, &m_gpu, &n_gpu, &work,
            &curand_state);
    TIMER_STOP("Finished station beam (3D)");
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy beam pattern back to CPU.
    oskar_Mem beam_pattern_cpu(&beam_pattern, OSKAR_LOCATION_CPU);

    // Save beam to file for plotting.
    const char* filename = "temp_test_beam_pattern.txt";
    FILE* file = fopen(filename, "w");
    for (int i = 0; i < num_pixels; ++i)
    {
        fprintf(file, "%10.3f,%10.3f,%10.3f,%10.3f\n",
                ((float*)l_cpu.data)[i],
                ((float*)m_cpu.data)[i],
                ((float2*)(beam_pattern_cpu.data))[i].x,
                ((float2*)(beam_pattern_cpu.data))[i].y);
    }
    fclose(file);

    /*--------------------------------------------------------------------------
        data = dlmread('temp_test_beam_pattern.txt');
        imagesc(log10(reshape(data(:,3), 401, 401).^2));
    --------------------------------------------------------------------------*/
}

void Test_evaluate_station_beam::performance_test()
{
}
