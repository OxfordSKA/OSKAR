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

#include <cuda_runtime_api.h>

#include "station/test/Test_evaluate_station_beam_dipoles.h"

#include "station/oskar_evaluate_array_pattern_dipoles_cuda.h"
#include "station/oskar_evaluate_station_beam.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"
#include "oskar_global.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define C_0 299792458.0

void Test_evaluate_station_beam_dipoles::test()
{
    double freq = 100e6;
    int num_az = 360;
    int num_el = 90;
    int num_pixels = num_az * num_el;
    oskar_Mem azimuth(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem elevation(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem l_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem m_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem n_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);

    // Generate azimuth and elevation.
    double* az = (double*)azimuth;
    double* el = (double*)elevation;
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
        ((double*)l_cpu)[i] = x;
        ((double*)m_cpu)[i] = y;
        ((double*)n_cpu)[i] = z;
    }

    // Copy to GPU.
    oskar_Mem l(&l_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem m(&m_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem n(&n_cpu, OSKAR_LOCATION_GPU);

    // Allocate GPU memory for result.
    oskar_Mem pattern(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_GPU, num_pixels);

    // Generate antenna array.
    int num_antennas_side = 100;
    int num_antennas = num_antennas_side * num_antennas_side;
    oskar_Mem antenna_x_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem antenna_y_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem antenna_z_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem cos_orn_x_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem sin_orn_x_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem cos_orn_y_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);
    oskar_Mem sin_orn_y_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_antennas);

    // Generate square array of antennas.
    const float sep = 0.5 * 2 * M_PI * freq / C_0; // Antenna separation, radians.
    const float half_array_size = (num_antennas_side - 1) * sep / 2.0;
    for (int x = 0; x < num_antennas_side; ++x)
    {
        for (int y = 0; y < num_antennas_side; ++y)
        {
            int i = y + x * num_antennas_side;

            // Antenna positions.
            ((double*)antenna_x_cpu)[i] = x * sep - half_array_size;
            ((double*)antenna_y_cpu)[i] = y * sep - half_array_size;
            ((double*)antenna_z_cpu)[i] = 0.0;

            // Antenna orientations.
            ((double*)cos_orn_x_cpu)[i] = cos(90.0 * M_PI / 180);
            ((double*)sin_orn_x_cpu)[i] = sin(90.0 * M_PI / 180);
            ((double*)cos_orn_y_cpu)[i] = cos(0.0 * M_PI / 180);
            ((double*)sin_orn_y_cpu)[i] = sin(0.0 * M_PI / 180);
        }
    }

    // TODO Generate the weights.
    oskar_Mem weights(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, num_antennas);

    // Copy all to GPU.
    oskar_Mem antenna_x(&antenna_x_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem antenna_y(&antenna_y_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem antenna_z(&antenna_z_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem cos_orientation_x(&cos_orn_x_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem sin_orientation_x(&sin_orn_x_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem cos_orientation_y(&cos_orn_y_cpu, OSKAR_LOCATION_GPU);
    oskar_Mem sin_orientation_y(&sin_orn_y_cpu, OSKAR_LOCATION_GPU);

    // Call the kernel.
    TIMER_START
    oskar_evaluate_array_pattern_dipoles_cuda_d (num_antennas,
            antenna_x, antenna_y, antenna_z, cos_orientation_x,
            sin_orientation_x, cos_orientation_y, sin_orientation_y,
            weights, num_pixels, l, m, n, pattern);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished station beam evaluation using dipoles (%d points)",
            num_pixels)

    cudaError_t err = cudaPeekAtLastError();
    CPPUNIT_ASSERT_EQUAL(0, (int) err);

    // Copy the memory back.
    oskar_Mem pattern_cpu(&pattern, OSKAR_LOCATION_CPU);

    const char filename[] = "cpp_unit_test_station_beam_dipoles.dat";
    FILE* file = fopen(filename, "w");
    double4c* p = (double4c*)pattern_cpu;
    for (int i = 0; i < num_pixels; ++i)
    {
        fprintf(file, "%f %f %f %f %f %f %f %f %f %f\n", az[i], el[i],
                p[i].a.x, p[i].a.y, p[i].b.x, p[i].b.y,
                p[i].c.x, p[i].c.y, p[i].d.x, p[i].d.y);
    }
    fclose(file);
}

