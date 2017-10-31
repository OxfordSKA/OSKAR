/*
 * Copyright (c) 2017, The University of Oxford
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

#include "math/oskar_dft_c2r.h"
#include "math/oskar_cmath.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_cl_utils.h"

#include <cstdlib>
#include <cstdio>

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
    oskar_mem_set_value_real(amp, 1.0, 0, 0, status);
    oskar_mem_set_value_real(wt, 1.0, 0, 0, status);

    /* Run DFT. */
    oskar_dft_c2r(num_baselines, wavenumber, u_, v_, w_, amp, wt,
            (int) num_pixels, l_, m_, 0, out, status);
    EXPECT_EQ(0, *status);

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
    int side = 128, status = 0;
    int type = OSKAR_SINGLE;
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
#ifdef OSKAR_HAVE_OPENCL
    oskar_cl_init("GPU", "NVIDIA|AMD");
    printf("Using %s\n", oskar_cl_device_name());
    run_test(type, OSKAR_CL, (int) num_baselines, u, v, w,
            side, l, m, n, "test_dft_cl", &status);
    oskar_cl_free();
#endif
#ifdef OSKAR_HAVE_CUDA
    run_test(type, OSKAR_GPU, (int) num_baselines, u, v, w,
            side, l, m, n, "test_dft_cuda", &status);
#endif
    run_test(type, OSKAR_CPU, (int) num_baselines, u, v, w,
            side, l, m, n, "test_dft_cpu", &status);

    oskar_mem_free(l, &status);
    oskar_mem_free(m, &status);
    oskar_mem_free(n, &status);
    oskar_mem_free(u, &status);
    oskar_mem_free(v, &status);
    oskar_mem_free(w, &status);
}
