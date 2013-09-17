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

#include "station/oskar_ElementModel.h"
#include "station/oskar_element_model_evaluate.h"
#include "station/test/Test_dipole_pattern.h"
#include <oskar_mem.h>
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"
#include "oskar_global.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

void Test_dipole_pattern::test()
{
    int num_az = 181;
    int num_el = 46;
    int num_pixels = num_az * num_el;
    int num_threads = 256;
    int num_blocks  = (num_pixels + num_threads - 1) / num_threads;
    oskar_Mem azimuth(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem elevation(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem l_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem m_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem n_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_pixels);

    // Generate azimuth and elevation.
    double* az = (double*)azimuth;
    double* el = (double*)elevation;
    for (int i = 0; i < num_az; ++i)
    {
        for (int j = 0; j < num_el; ++j)
        {
            az[j + i * num_el] = i * ((2.0 * M_PI) / (num_az - 1));
            el[j + i * num_el] = j * ((M_PI / 2.0) / (num_el - 1));
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

    // Define antenna orientations.
    double orientation_x = -120.0 * M_PI/180;
    double orientation_y = 137.0 * M_PI/180;

    // Call the kernel.
    TIMER_START
    oskar_cudak_evaluate_dipole_pattern_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_pixels, l, m, n,
            cos(orientation_x), sin(orientation_x),
            cos(orientation_y), sin(orientation_y),
            pattern);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished dipole pattern evaluation (%d points)", num_pixels)

    cudaError_t err = cudaPeekAtLastError();
    CPPUNIT_ASSERT_EQUAL(0, (int) err);

    // Copy the memory back.
    oskar_Mem pattern_cpu(&pattern, OSKAR_LOCATION_CPU);

    // New evaluation.
    oskar_ElementModel model;
    oskar_Mem theta(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_pixels);
    oskar_Mem phi(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_pixels);
    int error = oskar_element_model_evaluate(&model, &pattern, 1, orientation_x,
            orientation_y, num_pixels, &l, &m, &n, &theta, &phi);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

    // Copy the memory back.
    oskar_Mem pattern_cpu2(&pattern, OSKAR_LOCATION_CPU);
    for (int i = 0; i < num_pixels; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].a.x, ((double4c*)pattern_cpu2.data)[i].a.x, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].b.x, ((double4c*)pattern_cpu2.data)[i].b.x, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].c.x, ((double4c*)pattern_cpu2.data)[i].c.x, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].d.x, ((double4c*)pattern_cpu2.data)[i].d.x, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].a.y, ((double4c*)pattern_cpu2.data)[i].a.y, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].b.y, ((double4c*)pattern_cpu2.data)[i].b.y, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].c.y, ((double4c*)pattern_cpu2.data)[i].c.y, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)pattern_cpu.data)[i].d.y, ((double4c*)pattern_cpu2.data)[i].d.y, 1e-6);
    }

    const char filename[] = "cpp_unit_test_dipole_pattern.dat";
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
