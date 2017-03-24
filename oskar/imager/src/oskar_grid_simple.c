/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "imager/oskar_grid_simple.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_grid_simple_3_100_d(const double* restrict conv_func,
        const int num_vis, const double* restrict uu, const double* restrict vv,
        const double* restrict vis, const double* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        double* restrict grid);
static void oskar_grid_simple_3_100_f(const double* restrict conv_func,
        const int num_vis, const float* restrict uu, const float* restrict vv,
        const float* restrict vis, const float* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        float* restrict grid);

void oskar_grid_simple_d(const int support, const int oversample,
        const double* restrict conv_func, const int num_vis,
        const double* restrict uu, const double* restrict vv,
        const double* restrict vis, const double* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        double* restrict grid)
{
    size_t p;
    int i, j, k, grid_x, grid_y, off_x, off_y;
    const int g_centre = grid_size / 2;
    const double scale = grid_size * cell_size_rad;

    /* Use slightly more efficient version for default parameters. */
    if (support == 3 && oversample == 100)
    {
        oskar_grid_simple_3_100_d(conv_func, num_vis, uu, vv, vis,
                weight, cell_size_rad, grid_size, num_skipped, norm, grid);
        return;
    }

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        double cxy, cx, cy, pos_x, pos_y, val[2], sum, w;

        /* Convert UV coordinates to grid coordinates. */
        pos_x = -uu[i] * scale;
        pos_y = vv[i] * scale;
        grid_x = (int)round(pos_x) + g_centre;
        grid_y = (int)round(pos_y) + g_centre;

        /* Catch points that would lie outside the grid. */
        if (grid_x + support >= grid_size || grid_x - support < 0 ||
                grid_y + support >= grid_size || grid_y - support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_x = (int)round((grid_x - g_centre - pos_x) * oversample);
        off_y = (int)round((grid_y - g_centre - pos_y) * oversample);

        /* Convolve this point. */
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        for (j = -support; j <= support; ++j)
        {
            cy = conv_func[abs(off_y + j * oversample)];
            for (k = -support; k <= support; ++k)
            {
                cx = conv_func[abs(off_x + k * oversample)];
                cxy = cx * cy;
                sum += cxy;
                p = 2 * (((grid_y + j) * grid_size) + grid_x + k);
                grid[p]     += val[0] * cxy;
                grid[p + 1] += val[1] * cxy;
            }
        }
        *norm += sum * w;
    }
}


void oskar_grid_simple_f(const int support, const int oversample,
        const double* restrict conv_func, const int num_vis,
        const float* restrict uu, const float* restrict vv,
        const float* restrict vis, const float* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        float* restrict grid)
{
    size_t p;
    int i, j, k, grid_x, grid_y, off_x, off_y;
    const int g_centre = grid_size / 2;
    const double scale = grid_size * cell_size_rad;

    /* Use slightly more efficient version for default parameters. */
    if (support == 3 && oversample == 100)
    {
        oskar_grid_simple_3_100_f(conv_func, num_vis, uu, vv, vis,
                weight, cell_size_rad, grid_size, num_skipped, norm, grid);
        return;
    }

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        double cxy, cx, cy, pos_x, pos_y, val[2], sum, w;

        /* Convert UV coordinates to grid coordinates. */
        pos_x = -uu[i] * scale;
        pos_y = vv[i] * scale;
        grid_x = (int)round(pos_x) + g_centre;
        grid_y = (int)round(pos_y) + g_centre;

        /* Catch points that would lie outside the grid. */
        if (grid_x + support >= grid_size || grid_x - support < 0 ||
                grid_y + support >= grid_size || grid_y - support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_x = (int)round((grid_x - g_centre - pos_x) * oversample);
        off_y = (int)round((grid_y - g_centre - pos_y) * oversample);

        /* Convolve this point. */
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        for (j = -support; j <= support; ++j)
        {
            cy = conv_func[abs(off_y + j * oversample)];
            for (k = -support; k <= support; ++k)
            {
                cx = conv_func[abs(off_x + k * oversample)];
                cxy = cx * cy;
                sum += cxy;
                p = 2 * (((grid_y + j) * grid_size) + grid_x + k);
                grid[p]     += val[0] * cxy;
                grid[p + 1] += val[1] * cxy;
            }
        }
        *norm += sum * w;
    }
}


void oskar_grid_simple_3_100_d(const double* restrict conv_func,
        const int num_vis, const double* restrict uu, const double* restrict vv,
        const double* restrict vis, const double* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        double* restrict grid)
{
    size_t p;
    int i, j, k, grid_x, grid_y, off_x, off_y;
    const int g_centre = grid_size / 2;
    const double scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        double cxy, cx, cy, pos_x, pos_y, val[2], sum, w;

        /* Convert UV coordinates to grid coordinates. */
        pos_x = -uu[i] * scale;
        pos_y = vv[i] * scale;
        grid_x = (int)round(pos_x) + g_centre;
        grid_y = (int)round(pos_y) + g_centre;

        /* Catch points that would lie outside the grid. */
        if (grid_x + 3 >= grid_size || grid_x - 3 < 0 ||
                grid_y + 3 >= grid_size || grid_y - 3 < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_x = (int)round((grid_x - g_centre - pos_x) * 100);
        off_y = (int)round((grid_y - g_centre - pos_y) * 100);

        /* Convolve this point. */
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        for (j = -3; j <= 3; ++j)
        {
            cy = conv_func[abs(off_y + j * 100)];
            for (k = -3; k <= 3; ++k)
            {
                cx = conv_func[abs(off_x + k * 100)];
                cxy = cx * cy;
                sum += cxy;
                p = 2 * (((grid_y + j) * grid_size) + grid_x + k);
                grid[p]     += val[0] * cxy;
                grid[p + 1] += val[1] * cxy;
            }
        }
        *norm += sum * w;
    }
}


void oskar_grid_simple_3_100_f(const double* restrict conv_func,
        const int num_vis, const float* restrict uu, const float* restrict vv,
        const float* restrict vis, const float* restrict weight,
        const double cell_size_rad, const int grid_size,
        int* restrict num_skipped, double* restrict norm,
        float* restrict grid)
{
    size_t p;
    int i, j, k, grid_x, grid_y, off_x, off_y;
    const int g_centre = grid_size / 2;
    const double scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        double cxy, cx, cy, pos_x, pos_y, val[2], sum, w;

        /* Convert UV coordinates to grid coordinates. */
        pos_x = -uu[i] * scale;
        pos_y = vv[i] * scale;
        grid_x = (int)round(pos_x) + g_centre;
        grid_y = (int)round(pos_y) + g_centre;

        /* Catch points that would lie outside the grid. */
        if (grid_x + 3 >= grid_size || grid_x - 3 < 0 ||
                grid_y + 3 >= grid_size || grid_y - 3 < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_x = (int)round((grid_x - g_centre - pos_x) * 100);
        off_y = (int)round((grid_y - g_centre - pos_y) * 100);

        /* Convolve this point. */
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        for (j = -3; j <= 3; ++j)
        {
            cy = conv_func[abs(off_y + j * 100)];
            for (k = -3; k <= 3; ++k)
            {
                cx = conv_func[abs(off_x + k * 100)];
                cxy = cx * cy;
                sum += cxy;
                p = 2 * (((grid_y + j) * grid_size) + grid_x + k);
                grid[p]     += val[0] * cxy;
                grid[p + 1] += val[1] * cxy;
            }
        }
        *norm += sum * w;
    }
}

#ifdef __cplusplus
}
#endif
