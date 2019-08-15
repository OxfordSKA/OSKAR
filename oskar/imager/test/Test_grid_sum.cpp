/*
 * Copyright (c) 2016, The University of Oxford
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
#include "imager/oskar_imager.h"

// #define WRITE_FITS 1
#ifdef WRITE_FITS
#include <fitsio.h>
#endif

TEST(imager, grid_sum)
{
    int status = 0, type = OSKAR_DOUBLE;
    int size = 2048, grid_size = size * size;

    // Create and set up the imager.
    oskar_Imager* im = oskar_imager_create(type, &status);
    oskar_imager_set_grid_kernel(im, "pillbox", 1, 1, &status);
    oskar_imager_set_fov(im, 5.0);
    oskar_imager_set_size(im, size, &status);
    oskar_Mem* grid = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU,
            grid_size, &status);
    ASSERT_EQ(0, status);

    // Create visibility data.
    int num_vis = 10000;
    oskar_Mem* uu = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
    oskar_Mem* vv = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
    oskar_Mem* ww = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
    oskar_Mem* vis = oskar_mem_create(type | OSKAR_COMPLEX, OSKAR_CPU, num_vis,
            &status);
    oskar_Mem* weight = oskar_mem_create(type, OSKAR_CPU, num_vis, &status);
    oskar_mem_random_gaussian(uu, 0, 1, 2, 3, 100.0, &status);
    oskar_mem_random_gaussian(vv, 4, 5, 6, 7, 100.0, &status);
    oskar_mem_set_value_real(vis, 1.0, 0, num_vis, &status);
    oskar_mem_set_value_real(weight, 1.0, 0, num_vis, &status);

    // Grid visibility data.
    double plane_norm = 0.0;
    oskar_imager_update_plane(im, num_vis, uu, vv, ww, vis, weight,
            0, grid, &plane_norm, 0, &status);
    ASSERT_DOUBLE_EQ((double)num_vis, plane_norm);

    // Sum the grid.
    double2* t = oskar_mem_double2(grid, &status);
    double sum = 0.0;
    for (int i = 0; i < grid_size; i++) sum += t[i].x;
    ASSERT_DOUBLE_EQ((double)num_vis, sum);

    // Finalise the image.
    oskar_imager_finalise_plane(im, grid, plane_norm, &status);
    ASSERT_EQ(0, status);

#ifdef WRITE_FITS
    // Get the real part only.
    if (oskar_mem_precision(grid) == OSKAR_DOUBLE)
    {
        double *t = oskar_mem_double(grid, &status);
        for (int j = 0; j < grid_size; ++j) t[j] = t[2 * j];
    }
    else
    {
        float *t = oskar_mem_float(grid, &status);
        for (int j = 0; j < grid_size; ++j) t[j] = t[2 * j];
    }

    // Save the real part.
    fitsfile* f;
    long naxes[2] = {size, size}, firstpix[2] = {1, 1};
    fits_create_file(&f, "test_imager_grid_sum.fits", &status);
    fits_create_img(f, (type == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG),
            2, naxes, &status);
    fits_write_pix(f, (type == OSKAR_DOUBLE ? TDOUBLE : TFLOAT),
            firstpix, grid_size, oskar_mem_void(grid), &status);
    fits_close_file(f, &status);
#endif

    // Clean up.
    oskar_imager_free(im, &status);
    oskar_mem_free(uu, &status);
    oskar_mem_free(vv, &status);
    oskar_mem_free(ww, &status);
    oskar_mem_free(vis, &status);
    oskar_mem_free(weight, &status);
    oskar_mem_free(grid, &status);
}
