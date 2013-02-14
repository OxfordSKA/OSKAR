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

#include "station/test/Test_evaluate_station_beam.h"

#include "oskar_global.h"
#include "station/oskar_evaluate_station_beam_aperture_array.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_station_model_save_config.h"
#include "station/oskar_station_model_resize_element_types.h"
#include "station/oskar_evaluate_station_beam_gaussian.h"
#include "station/oskar_station_model_multiply_by_wavenumber.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "utility/oskar_curand_state_free.h"
#include "utility/oskar_curand_state_init.h"
#include "utility/oskar_mem_binary_file_write.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"

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
    oskar_station_model_resize_element_types(&station_cpu, 1, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

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
    station_cpu.beam_longitude_rad  = 0.0;
    station_cpu.beam_latitude_rad = M_PI_2;

    // Set the station meta-data.
    station_cpu.element_pattern->type = OSKAR_ELEMENT_MODEL_TYPE_ISOTROPIC;
    station_cpu.use_polarised_elements = false;

    //    error = oskar_station_model_save_configuration("temp_test_station.txt", &station_cpu);
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy the station structure to the gpu and scale the coordinates to wavenumbers.
    oskar_StationModel station_gpu(&station_cpu, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_METRES, station_gpu.coord_units);
    oskar_station_model_multiply_by_wavenumber(&station_gpu, frequency, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Evaluate horizontal l,m positions at which to generate the beam pattern.
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
    oskar_CurandState curand_state;
    int seed = 0;
    oskar_curand_state_init(&curand_state, num_antennas, seed, 0, 0, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    station_gpu.apply_element_errors = OSKAR_FALSE;

    // Allocate weights work array.
    oskar_WorkStationBeam work(OSKAR_SINGLE, OSKAR_LOCATION_GPU);

    // Declare memory for the beam pattern.
    oskar_Mem beam_pattern(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, num_pixels);

    station_gpu.array_is_3d = 0;
    TIMER_START
    oskar_evaluate_station_beam_aperture_array(&beam_pattern, &station_gpu,
            num_pixels, &l_gpu, &m_gpu, &n_gpu, gast, &work, &curand_state,
            &error);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished aperture array station beam (2D)");
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
    station_gpu.array_is_3d = 1;
    TIMER_START
    oskar_evaluate_station_beam_aperture_array(&beam_pattern, &station_gpu,
            num_pixels, &l_gpu, &m_gpu, &n_gpu, gast, &work, &curand_state,
            &error);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished aperture array station beam (3D)");
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
    oskar_curand_state_free(&curand_state, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);
}


void Test_evaluate_station_beam::evaluate_gaussian_pattern()
{
    int err = 0;
    double fwhm = 1.0;
    int size = 512;
    int num_points = size * size;
    double lm_minmax = 0.2;
    bool save_results = true;

    // Double CPU
    {
        int type = OSKAR_DOUBLE;
        int location = OSKAR_LOCATION_CPU;
        oskar_Mem x(type, location, size);
        oskar_linspace_d((double*)x.data, -lm_minmax, lm_minmax, size);
        oskar_Mem l(type, location, num_points);
        oskar_Mem m(type, location, num_points);
        oskar_Mem horizon_mask(type, location, num_points);
        oskar_meshgrid_d((double*)l.data, (double*)m.data,
                (double*)x.data, size, (double*)x.data, size);
        oskar_Mem beam(type | OSKAR_COMPLEX, location, num_points);

        oskar_evaluate_station_beam_gaussian(&beam, num_points, &l, &m,
                &horizon_mask, fwhm, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_beam_double_cpu.dat";
            remove(filename);
            oskar_mem_binary_file_write(&beam, filename, 0, 0, 0, beam.num_elements, &err);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        }
    }

    // Single CPU
    {
        int type = OSKAR_SINGLE;
        int location = OSKAR_LOCATION_CPU;
        oskar_Mem x(type, location, size);
        oskar_linspace_f((float*)x.data, -lm_minmax, lm_minmax, size);
        oskar_Mem l(type, location, num_points);
        oskar_Mem m(type, location, num_points);
        oskar_Mem horizon_mask(type, location, num_points);
        oskar_meshgrid_f((float*)l.data, (float*)m.data,
                (float*)x.data, size, (float*)x.data, size);
        oskar_Mem beam(type | OSKAR_COMPLEX, location, num_points);

        oskar_evaluate_station_beam_gaussian(&beam, num_points, &l, &m,
                &horizon_mask, fwhm, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_beam_single_cpu.dat";
            remove(filename);
            oskar_mem_binary_file_write(&beam, filename, 0, 0, 0, beam.num_elements, &err);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        }
    }

    // Double GPU
    {
        int type = OSKAR_DOUBLE;
        int location = OSKAR_LOCATION_GPU;

        oskar_Mem xcpu(type, OSKAR_LOCATION_CPU, size);
        oskar_linspace_d((double*)xcpu.data, -lm_minmax, lm_minmax, size);
        oskar_Mem lcpu(type, OSKAR_LOCATION_CPU, num_points);
        oskar_Mem mcpu(type, OSKAR_LOCATION_CPU, num_points);
        oskar_meshgrid_d((double*)lcpu.data, (double*)mcpu.data,
                (double*)xcpu.data, size, (double*)xcpu.data, size);

        oskar_Mem l(&lcpu, location);
        oskar_Mem m(&mcpu, location);
        oskar_Mem horizon_mask(type, location, num_points);
        oskar_Mem beam(type | OSKAR_COMPLEX, location, num_points);

        oskar_evaluate_station_beam_gaussian(&beam, num_points, &l, &m,
                &horizon_mask, fwhm, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_beam_double_gpu.dat";
            remove(filename);
            oskar_mem_binary_file_write(&beam, filename, 0, 0, 0, beam.num_elements, &err);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        }
    }

    // Single GPU
    {
        int type = OSKAR_SINGLE;
        int location = OSKAR_LOCATION_GPU;

        oskar_Mem xcpu(type, OSKAR_LOCATION_CPU, size);
        oskar_linspace_f((float*)xcpu.data, -lm_minmax, lm_minmax, size);
        oskar_Mem lcpu(type, OSKAR_LOCATION_CPU, num_points);
        oskar_Mem mcpu(type, OSKAR_LOCATION_CPU, num_points);
        oskar_meshgrid_f((float*)lcpu.data, (float*)mcpu.data,
                (float*)xcpu.data, size, (float*)xcpu.data, size);

        oskar_Mem l(&lcpu, location);
        oskar_Mem m(&mcpu, location);
        oskar_Mem horizon_mask(type, location, num_points);
        oskar_Mem beam(type | OSKAR_COMPLEX, location, num_points);

        oskar_evaluate_station_beam_gaussian(&beam, num_points, &l, &m,
                &horizon_mask, fwhm, &err);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

        // Write output to file.
        if (save_results)
        {
            const char* filename = "temp_beam_single_gpu.dat";
            remove(filename);
            oskar_mem_binary_file_write(&beam, filename, 0, 0, 0, beam.num_elements, &err);
            CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
        }
    }

}
