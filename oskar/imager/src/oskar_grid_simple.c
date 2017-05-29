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

#define D_SUPPORT 3
#define D_OVERSAMPLE 100

static void oskar_grid_simple_default_d(
        const double* restrict conv_func,
        const size_t num_points,
        const double* restrict uu,
        const double* restrict vv,
        const double* restrict vis,
        const double* restrict weight,
        const double cell_size_rad,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        double* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const double pos_u = -uu[i] * grid_scale;
        const double pos_v = vv[i] * grid_scale;
        const int grid_u = (int)round(pos_u) + grid_centre;
        const int grid_v = (int)round(pos_v) + grid_centre;

        /* Get visibility data. */
        const double weight_i = weight[i];
        const double v_re = weight_i * vis[2 * i];
        const double v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)round((round(pos_u) - pos_u) * D_OVERSAMPLE);
        const int off_v = (int)round((round(pos_v) - pos_v) * D_OVERSAMPLE);

        /* Catch points that would lie outside the grid. */
        if (grid_u + D_SUPPORT >= grid_size || grid_u - D_SUPPORT < 0 ||
                grid_v + D_SUPPORT >= grid_size || grid_v - D_SUPPORT < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -D_SUPPORT; j <= D_SUPPORT; ++j)
        {
            size_t p1;
            const double c1 = conv_func[abs(off_v + j * D_OVERSAMPLE)];
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -D_SUPPORT; k <= D_SUPPORT; ++k)
            {
                const size_t p = (p1 + k) << 1;
                const double c = conv_func[abs(off_u + k * D_OVERSAMPLE)] * c1;
                grid[p]     += v_re * c;
                grid[p + 1] += v_im * c;
                sum += c;
            }
        }
        *norm += sum * weight_i;
    }
}


static void oskar_grid_simple_default_f(
        const float* restrict conv_func,
        const size_t num_points,
        const float* restrict uu,
        const float* restrict vv,
        const float* restrict vis,
        const float* restrict weight,
        const float cell_size_rad,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        float* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const int grid_u = (int)roundf(pos_u) + grid_centre;
        const int grid_v = (int)roundf(pos_v) + grid_centre;

        /* Get visibility data. */
        const float weight_i = weight[i];
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)roundf((roundf(pos_u) - pos_u) * D_OVERSAMPLE);
        const int off_v = (int)roundf((roundf(pos_v) - pos_v) * D_OVERSAMPLE);

        /* Catch points that would lie outside the grid. */
        if (grid_u + D_SUPPORT >= grid_size || grid_u - D_SUPPORT < 0 ||
                grid_v + D_SUPPORT >= grid_size || grid_v - D_SUPPORT < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -D_SUPPORT; j <= D_SUPPORT; ++j)
        {
            size_t p1;
            const float c1 = conv_func[abs(off_v + j * D_OVERSAMPLE)];
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -D_SUPPORT; k <= D_SUPPORT; ++k)
            {
                const size_t p = (p1 + k) << 1;
                const float c = conv_func[abs(off_u + k * D_OVERSAMPLE)] * c1;
                grid[p]     += v_re * c;
                grid[p + 1] += v_im * c;
                sum += c;
            }
        }
        *norm += sum * weight_i;
    }
}


void oskar_grid_simple_d(
        const int support,
        const int oversample,
        const double* restrict conv_func,
        const size_t num_points,
        const double* restrict uu,
        const double* restrict vv,
        const double* restrict vis,
        const double* restrict weight,
        const double cell_size_rad,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        double* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Use slightly more efficient version for default parameters. */
    if (support == D_SUPPORT && oversample == D_OVERSAMPLE)
    {
        oskar_grid_simple_default_d(conv_func, num_points, uu, vv, vis,
                weight, cell_size_rad, grid_size, num_skipped, norm, grid);
        return;
    }

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const double pos_u = -uu[i] * grid_scale;
        const double pos_v = vv[i] * grid_scale;
        const int grid_u = (int)round(pos_u) + grid_centre;
        const int grid_v = (int)round(pos_v) + grid_centre;

        /* Get visibility data. */
        const double weight_i = weight[i];
        const double v_re = weight_i * vis[2 * i];
        const double v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)round((round(pos_u) - pos_u) * oversample);
        const int off_v = (int)round((round(pos_v) - pos_v) * oversample);

        /* Catch points that would lie outside the grid. */
        if (grid_u + support >= grid_size || grid_u - support < 0 ||
                grid_v + support >= grid_size || grid_v - support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -support; j <= support; ++j)
        {
            size_t p1;
            const double c1 = conv_func[abs(off_v + j * oversample)];
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -support; k <= support; ++k)
            {
                const size_t p = (p1 + k) << 1;
                const double c = conv_func[abs(off_u + k * oversample)] * c1;
                grid[p]     += v_re * c;
                grid[p + 1] += v_im * c;
                sum += c;
            }
        }
        *norm += sum * weight_i;
    }
}


void oskar_grid_simple_f(
        const int support,
        const int oversample,
        const float* restrict conv_func,
        const size_t num_points,
        const float* restrict uu,
        const float* restrict vv,
        const float* restrict vis,
        const float* restrict weight,
        const float cell_size_rad,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        float* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Use slightly more efficient version for default parameters. */
    if (support == D_SUPPORT && oversample == D_OVERSAMPLE)
    {
        oskar_grid_simple_default_f(conv_func, num_points, uu, vv, vis,
                weight, cell_size_rad, grid_size, num_skipped, norm, grid);
        return;
    }

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const int grid_u = (int)roundf(pos_u) + grid_centre;
        const int grid_v = (int)roundf(pos_v) + grid_centre;

        /* Get visibility data. */
        const float weight_i = weight[i];
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)roundf((roundf(pos_u) - pos_u) * oversample);
        const int off_v = (int)roundf((roundf(pos_v) - pos_v) * oversample);

        /* Catch points that would lie outside the grid. */
        if (grid_u + support >= grid_size || grid_u - support < 0 ||
                grid_v + support >= grid_size || grid_v - support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -support; j <= support; ++j)
        {
            size_t p1;
            const float c1 = conv_func[abs(off_v + j * oversample)];
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -support; k <= support; ++k)
            {
                const size_t p = (p1 + k) << 1;
                const float c = conv_func[abs(off_u + k * oversample)] * c1;
                grid[p]     += v_re * c;
                grid[p + 1] += v_im * c;
                sum += c;
            }
        }
        *norm += sum * weight_i;
    }
}

#ifdef __cplusplus
}
#endif
