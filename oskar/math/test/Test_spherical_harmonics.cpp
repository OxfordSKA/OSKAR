/*
 * Copyright (c) 2019, The University of Oxford
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

#include "math/oskar_cmath.h"
#include "math/oskar_spherical_harmonic.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"

#include <cstdio>
#include <string>

static void test_sum(int type, int location, int side_length, int l_max,
        const oskar_Mem* theta, const oskar_Mem* phi, const char* root_name)
{
    int status = 0;
    const int num_pixels = side_length * side_length;
    const int num_coeff = (l_max + 1) * (l_max + 1);

    /* Copy data to device. */
    oskar_Mem* theta_t = oskar_mem_create_copy(theta, location, &status);
    oskar_Mem* phi_t = oskar_mem_create_copy(phi, location, &status);
    oskar_Mem* coeff = oskar_mem_create(type, location, num_coeff, &status);
    oskar_Mem* surface = oskar_mem_create(type, location,
            num_pixels * num_coeff, &status);
    oskar_SphericalHarmonic* sh = oskar_spherical_harmonic_create(
            type, location, l_max, &status);

    /* Spherical harmonic sum (only one term at a time). */
    oskar_Timer* tmr = oskar_timer_create(location);
    for (int l = 0; l <= l_max; ++l)
    {
        for (int m = -l; m <= l; ++m)
        {
            oskar_mem_clear_contents(coeff, &status);
            const int coeff_idx = l*l + (m + l);
            oskar_mem_set_element_real(coeff, coeff_idx, 1.0, &status);
            //oskar_mem_set_value_real(coeff, 1.0, 0, num_coeff, &status);
            oskar_spherical_harmonic_set_coeff(sh, l_max, coeff, &status);
            const int offset = coeff_idx * num_pixels;
            oskar_timer_resume(tmr);
            oskar_spherical_harmonic_sum(sh, num_pixels, theta_t, phi_t,
                    1, offset, surface, &status);
            oskar_timer_pause(tmr);
        }
    }
    printf("Time taken for oskar_spherical_harmonic_sum(): %.3f sec\n",
            oskar_timer_elapsed(tmr));
    EXPECT_EQ(0, status) << oskar_get_error_string(status);

    /* Write data. */
    oskar_mem_write_fits_cube(surface, root_name,
            side_length, side_length, num_coeff, -1, &status);

    /* Free memory. */
    oskar_timer_free(tmr);
    oskar_spherical_harmonic_free(sh);
    oskar_mem_free(theta_t, &status);
    oskar_mem_free(phi_t, &status);
    oskar_mem_free(surface, &status);
    oskar_mem_free(coeff, &status);
}

#define GENERATE_GRID(FP) \
        for (int y = 0; y < side; ++y) {\
            const double m = -1.0 + y * cell;\
            for (int x = 0; x < side; ++x) {\
                const double l = -1.0 + x * cell;\
                const double r = sqrt(l*l + m*m);\
                const int i = x + y * side;\
                if (r > 1.0) {\
                    t_[i] = (FP)sqrt(-1.0);\
                    p_[i] = (FP)sqrt(-1.0);\
                }\
                else {\
                    t_[i] = (FP)asin(r);\
                    p_[i] = (FP)atan2(m, l);\
                }\
            }\
        }\

TEST(spherical_harmonics, sum)
{
    int num_devices = 0, side = 128, status = 0;
    int type = OSKAR_SINGLE, location = 0;
    int l_max = 17; /* Maximum in single precision is 17. */
    int num_pixels = side * side;
    double cell = 2.0 / (side - 1);

    /* Generate input data. */
    oskar_Mem* theta = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
    oskar_Mem* phi = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
    if (type == OSKAR_SINGLE)
    {
        float* t_ = oskar_mem_float(theta, &status);
        float* p_ = oskar_mem_float(phi, &status);
        GENERATE_GRID(float)
    }
    else
    {
        double* t_ = oskar_mem_double(theta, &status);
        double* p_ = oskar_mem_double(phi, &status);
        GENERATE_GRID(double)
    }

    /* Run on devices. */
    oskar_device_set_require_double_precision(0);
    for (int r = 0; r < 1; ++r)
    {
        num_devices = oskar_device_count("CUDA", &location);
        for (int i = 0; i < num_devices; ++i)
        {
            oskar_device_set(location, i, &status);
            char* device_name = oskar_device_name(location, i);
            std::string device(device_name);
            free(device_name);
            printf("Using CUDA device %s\n", device.c_str());
            for (size_t j = 0; j < device.length(); ++j)
                if (device[j] == ' ') device[j] = '_';
            device = std::string("test_spherical_harmonic_sum_cuda_") + device;
            test_sum(type, location, side, l_max, theta, phi, device.c_str());
        }
    }
    for (int r = 0; r < 1; ++r)
    {
        num_devices = oskar_device_count("OpenCL", &location);
        for (int i = 0; i < num_devices; ++i)
        {
            oskar_device_set(location, i, &status);
            char* device_name = oskar_device_name(location, i);
            std::string device(device_name);
            free(device_name);
            printf("Using OpenCL device %s\n", device.c_str());
            for (size_t j = 0; j < device.length(); ++j)
                if (device[j] == ' ' || device[j] == '@' ||
                        device[j] == '(' || device[j] == ')')
                    device[j] = '_';
            device = std::string("test_spherical_harmonic_sum_cl_") + device;
            test_sum(type, location, side, l_max, theta, phi, device.c_str());
        }
    }
    test_sum(type, OSKAR_CPU, side, l_max, theta, phi,
            "test_spherical_harmonic_sum_cpu");

    oskar_mem_free(theta, &status);
    oskar_mem_free(phi, &status);
}
