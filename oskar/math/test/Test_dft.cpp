/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "math/oskar_dft_c2r.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"

#include <cstdlib>
#include <cstdio>
#include <string>

static void run_test(int type, int loc, int num_baselines,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w, int side,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const char* filename, int* status)
{
    double freq = 100e6;
    double wavenumber = 2 * M_PI * freq / 299792458.;
    size_t num_pixels = side * side;

    /* Copy data to device. */
    oskar_Mem* l_ = oskar_mem_create_copy(l, loc, status);
    oskar_Mem* m_ = oskar_mem_create_copy(m, loc, status);
    oskar_Mem* n_ = oskar_mem_create_copy(n, loc, status);
    oskar_Mem* u_ = oskar_mem_create_copy(u, loc, status);
    oskar_Mem* v_ = oskar_mem_create_copy(v, loc, status);
    oskar_Mem* w_ = oskar_mem_create_copy(w, loc, status);
    oskar_Mem *out = oskar_mem_create(type, loc, num_pixels, status);
    oskar_Mem *amp = oskar_mem_create(type | OSKAR_COMPLEX,
            loc, num_baselines, status);
    oskar_Mem *wt = oskar_mem_create(type, loc, num_baselines, status);
    oskar_mem_set_value_real(amp, 1.0, 0, num_baselines, status);
    oskar_mem_set_value_real(wt, 1.0, 0, num_baselines, status);
    oskar_mem_clear_contents(out, status);

    /* Run DFT. */
    oskar_Timer* tmr = oskar_timer_create(loc);
    oskar_timer_resume(tmr);
    oskar_dft_c2r(num_baselines, wavenumber, u_, v_, w_, amp, wt,
            (int) num_pixels, l_, m_, 0, out, status);
    printf("Time taken for oskar_dft_c2r(): %.3f\n", oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);
    EXPECT_EQ(0, *status) << oskar_get_error_string(*status);

    /* Write data. */
    oskar_mem_write_fits_cube(out, filename, side, side, 1, 0, status);

    /* Free memory. */
    oskar_mem_free(l_, status);
    oskar_mem_free(m_, status);
    oskar_mem_free(n_, status);
    oskar_mem_free(u_, status);
    oskar_mem_free(v_, status);
    oskar_mem_free(w_, status);
    oskar_mem_free(amp, status);
    oskar_mem_free(wt, status);
    oskar_mem_free(out, status);
}

TEST(dft, c2r)
{
    int num_devices = 0, side = 128, status = 0;
    int type = OSKAR_SINGLE, location = 0;
    size_t num_pixels = side * side;
    size_t num_baselines = 1000;
    double fov = 4.0 * M_PI / 180.0;
    oskar_Mem *l = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
    oskar_Mem *m = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
    oskar_Mem *n = oskar_mem_create(type, OSKAR_CPU, num_pixels, &status);
    oskar_Mem *u = oskar_mem_create(type, OSKAR_CPU, num_baselines, &status);
    oskar_Mem *v = oskar_mem_create(type, OSKAR_CPU, num_baselines, &status);
    oskar_Mem *w = oskar_mem_create(type, OSKAR_CPU, num_baselines, &status);

    /* Generate input data. */
    oskar_evaluate_image_lmn_grid(side, side, fov, fov, 0, l, m, n, &status);
    oskar_mem_random_range(u, -1000., 1000., &status);
    oskar_mem_random_range(v, -1000., 1000., &status);
    oskar_mem_random_range(w, -200., 200., &status);
    ASSERT_EQ(0, status);

    /* Run on devices. */
    oskar_device_set_require_double_precision(0);
    num_devices = oskar_device_count("OpenCL", &location);
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_set(location, i, &status);
        char* device_name = oskar_device_name(location, i);
        std::string device(device_name);
        free(device_name);
        printf("Using OpenCL device %s\n", device.c_str());
        for (size_t j = 0; j < device.length(); ++j)
        {
            if (device[j] == ' ' || device[j] == '@' ||
                    device[j] == '(' || device[j] == ')')
            {
                device[j] = '_';
            }
        }
        device = std::string("test_dft_cl_") + device;
        run_test(type, location, (int) num_baselines, u, v, w,
                side, l, m, n, device.c_str(), &status);
    }
    num_devices = oskar_device_count("CUDA", &location);
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_set(location, i, &status);
        char* device_name = oskar_device_name(location, i);
        std::string device(device_name);
        free(device_name);
        printf("Using CUDA device %s\n", device.c_str());
        for (size_t j = 0; j < device.length(); ++j)
        {
            if (device[j] == ' ') device[j] = '_';
        }
        device = std::string("test_dft_cuda_") + device;
        run_test(type, location, (int) num_baselines, u, v, w,
                side, l, m, n, device.c_str(), &status);
    }
    run_test(type, OSKAR_CPU, (int) num_baselines, u, v, w,
            side, l, m, n, "test_dft_cpu", &status);

    oskar_mem_free(l, &status);
    oskar_mem_free(m, &status);
    oskar_mem_free(n, &status);
    oskar_mem_free(u, &status);
    oskar_mem_free(v, &status);
    oskar_mem_free(w, &status);
}
