/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_evaluate_array_pattern_dipoles_cuda.h>
#include <oskar_mem.h>
#include <oskar_cuda_check_error.h>

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(evaluate_station_beam_dipoles, test)
{
    double freq = 100e6;
    int num_az = 36;
    int num_el = 9;
    int num_pixels = num_az * num_el;
    int status = 0;
    oskar_Mem *h_az, *h_el, *d_l, *d_m, *d_n;

    {
        oskar_Mem *h_l, *h_m, *h_n;
        h_az = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels,
                &status);
        h_el = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels,
                &status);
        h_l = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels,
                &status);
        h_m = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels,
                &status);
        h_n = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels,
                &status);

        // Generate azimuth and elevation.
        double* az = oskar_mem_double(h_az, &status);
        double* el = oskar_mem_double(h_el, &status);
        double* l = oskar_mem_double(h_l, &status);
        double* m = oskar_mem_double(h_m, &status);
        double* n = oskar_mem_double(h_n, &status);
        for (int j = 0; j < num_el; ++j)
        {
            for (int i = 0; i < num_az; ++i)
            {
                az[i + j * num_az] = i * ((2.0 * M_PI) / (num_az - 1));
                el[i + j * num_az] = j * ((M_PI / 2.0) / (num_el - 1));
            }
        }

        // Convert to direction cosines.
        for (int i = 0; i < num_pixels; ++i)
        {
            double x, y, z;
            double cos_el = cos(el[i]);
            x = cos_el * sin(az[i]);
            y = cos_el * cos(az[i]);
            z = sin(el[i]);
            l[i] = x;
            m[i] = y;
            n[i] = z;
        }

        // Copy to GPU and free host memory.
        d_l = oskar_mem_create_copy(h_l, OSKAR_LOCATION_GPU, &status);
        d_m = oskar_mem_create_copy(h_m, OSKAR_LOCATION_GPU, &status);
        d_n = oskar_mem_create_copy(h_n, OSKAR_LOCATION_GPU, &status);
        oskar_mem_free(h_l, &status);
        oskar_mem_free(h_m, &status);
        oskar_mem_free(h_n, &status);
    }

    // Generate antenna array.
    int num_antennas_side = 100;
    int num_antennas = num_antennas_side * num_antennas_side;
    oskar_Mem *h_x, *h_y;
    h_x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas,
            &status);
    h_y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas,
            &status);

    // Generate square array of antennas.
    const float sep = 0.5; // Antenna separation, metres.
    const float half_array_size = (num_antennas_side - 1) * sep / 2.0;
    for (int x = 0; x < num_antennas_side; ++x)
    {
        for (int y = 0; y < num_antennas_side; ++y)
        {
            int i = y + x * num_antennas_side;

            // Antenna positions.
            oskar_mem_double(h_x, &status)[i] = x * sep - half_array_size;
            oskar_mem_double(h_y, &status)[i] = y * sep - half_array_size;
        }
    }

    // Copy to GPU and free host memory.
    oskar_Mem *d_x, *d_y, *d_z, *cos_x, *sin_x, *cos_y, *sin_y;
    oskar_Mem *weights, *pattern;
    d_z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    cos_x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    sin_x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    cos_y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    sin_y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    weights = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU,
            num_antennas, &status);
    pattern = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU,
            num_pixels, &status);
    oskar_mem_set_value_real(d_z, 0.0, 0, 0, &status);
    oskar_mem_set_value_real(cos_x, 0.0, 0, 0, &status);
    oskar_mem_set_value_real(sin_x, 1.0, 0, 0, &status);
    oskar_mem_set_value_real(cos_y, 1.0, 0, 0, &status);
    oskar_mem_set_value_real(sin_y, 0.0, 0, 0, &status);
    d_x = oskar_mem_create_copy(h_x, OSKAR_LOCATION_GPU, &status);
    d_y = oskar_mem_create_copy(h_y, OSKAR_LOCATION_GPU, &status);
    oskar_mem_free(h_x, &status);
    oskar_mem_free(h_y, &status);

    // Call the kernel.
    double wavenumber = 2.0 * M_PI * freq / 299792458.0;
    oskar_evaluate_array_pattern_dipoles_cuda_d (num_antennas, wavenumber,
            oskar_mem_double_const(d_x, &status),
            oskar_mem_double_const(d_y, &status),
            oskar_mem_double_const(d_z, &status),
            oskar_mem_double_const(cos_x, &status),
            oskar_mem_double_const(sin_x, &status),
            oskar_mem_double_const(cos_y, &status),
            oskar_mem_double_const(sin_y, &status),
            oskar_mem_double2_const(weights, &status), num_pixels,
            oskar_mem_double_const(d_l, &status),
            oskar_mem_double_const(d_m, &status),
            oskar_mem_double_const(d_n, &status),
            oskar_mem_double4c(pattern, &status));
    oskar_cuda_check_error(&status);
    ASSERT_EQ(0, status);

    const char* filename = "temp_test_station_beam_dipoles.dat";
    FILE* file = fopen(filename, "w");
    oskar_mem_save_ascii(file, 3, num_pixels, &status, h_az, h_el, pattern);
    fclose(file);
    remove(filename);

    oskar_mem_free(pattern, &status);
    oskar_mem_free(d_x, &status);
    oskar_mem_free(d_y, &status);
    oskar_mem_free(d_z, &status);
    oskar_mem_free(cos_x, &status);
    oskar_mem_free(sin_x, &status);
    oskar_mem_free(cos_y, &status);
    oskar_mem_free(sin_y, &status);
    oskar_mem_free(weights, &status);

    oskar_mem_free(h_az, &status);
    oskar_mem_free(h_el, &status);
    oskar_mem_free(d_l, &status);
    oskar_mem_free(d_m, &status);
    oskar_mem_free(d_n, &status);
}

