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

#include "imager/oskar_grid_wproj.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_wproj_d(const int num_w_planes, const int* restrict support,
        const int oversample, const int conv_size_half,
        const double* restrict conv_func, const int num_vis,
        const double* restrict uu, const double* restrict vv,
        const double* restrict ww, const double* restrict vis,
        const double* restrict weight, const double cell_size_rad,
        const double w_scale, const int grid_size, int* restrict num_skipped,
        double* restrict norm, double* restrict grid)
{
    int i;
    const int g_centre = grid_size / 2;
    const int kernel_dim = conv_size_half * conv_size_half;
    const double scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        size_t p;
        double cwt[2], pos_u, pos_v, val[2], sum, w, ww_i;
        int ix, iy, j, k, grid_u, grid_v, grid_w, off_u, off_v, wsupport;
        int kernel_start;

        /* Convert UV coordinates to grid coordinates. */
        pos_u = -uu[i] * scale;
        pos_v = vv[i] * scale;
        ww_i = ww[i];
        grid_u = (int)round(pos_u) + g_centre;
        grid_v = (int)round(pos_v) + g_centre;
        grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
        if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

        /* Catch points that would lie outside the grid. */
        wsupport = support[grid_w];
        if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
                grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_u = (int)round((grid_u - g_centre - pos_u) * oversample);
        off_v = (int)round((grid_v - g_centre - pos_v) * oversample);

        /* Convolve this point. */
        kernel_start = grid_w * kernel_dim;
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        if (ww_i > 0.0)
        {
            for (j = -wsupport; j <= wsupport; ++j)
            {
                iy = abs(off_v + j * oversample);
                for (k = -wsupport; k <= wsupport; ++k)
                {
                    ix = abs(off_u + k * oversample);
                    p = 2 * (kernel_start + iy * conv_size_half + ix);
                    cwt[0] = conv_func[p];
                    cwt[1] = -conv_func[p + 1]; /* Conjugate. */
                    sum += cwt[0]; /* Real part only. */
                    p = 2 * (((grid_v + j) * grid_size) + grid_u + k);
                    grid[p]     += (val[0] * cwt[0] - val[1] * cwt[1]);
                    grid[p + 1] += (val[1] * cwt[0] + val[0] * cwt[1]);
                }
            }
        }
        else
        {
            for (j = -wsupport; j <= wsupport; ++j)
            {
                iy = abs(off_v + j * oversample);
                for (k = -wsupport; k <= wsupport; ++k)
                {
                    ix = abs(off_u + k * oversample);
                    p = 2 * (kernel_start + iy * conv_size_half + ix);
                    cwt[0] = conv_func[p];
                    cwt[1] = conv_func[p + 1];
                    sum += cwt[0]; /* Real part only. */
                    p = 2 * (((grid_v + j) * grid_size) + grid_u + k);
                    grid[p]     += (val[0] * cwt[0] - val[1] * cwt[1]);
                    grid[p + 1] += (val[1] * cwt[0] + val[0] * cwt[1]);
                }
            }
        }
        *norm += sum * w;
    }
}


void oskar_grid_wproj_f(const int num_w_planes, const int* restrict support,
        const int oversample, const int conv_size_half,
        const float* restrict conv_func, const int num_vis,
        const float* restrict uu, const float* restrict vv,
        const float* restrict ww, const float* restrict vis,
        const float* restrict weight, const double cell_size_rad,
        const double w_scale, const int grid_size, int* restrict num_skipped,
        double* restrict norm, float* restrict grid)
{
    int i;
    const int g_centre = grid_size / 2;
    const int kernel_dim = conv_size_half * conv_size_half;
    const double scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_vis; ++i)
    {
        size_t p;
        double cwt[2], pos_u, pos_v, val[2], sum, w, ww_i;
        int ix, iy, j, k, grid_u, grid_v, grid_w, off_u, off_v, wsupport;
        int kernel_start;

        /* Convert UV coordinates to grid coordinates. */
        pos_u = -uu[i] * scale;
        pos_v = vv[i] * scale;
        ww_i = ww[i];
        grid_u = (int)round(pos_u) + g_centre;
        grid_v = (int)round(pos_v) + g_centre;
        grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
        if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

        /* Catch points that would lie outside the grid. */
        wsupport = support[grid_w];
        if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
                grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Scaled distance from nearest grid point. */
        off_u = (int)round((grid_u - g_centre - pos_u) * oversample);
        off_v = (int)round((grid_v - g_centre - pos_v) * oversample);

        /* Convolve this point. */
        kernel_start = grid_w * kernel_dim;
        w = weight[i];
        val[0] = w * vis[2 * i];
        val[1] = w * vis[2 * i + 1];
        sum = 0.0;
        if (ww_i > 0.0)
        {
            for (j = -wsupport; j <= wsupport; ++j)
            {
                iy = abs(off_v + j * oversample);
                for (k = -wsupport; k <= wsupport; ++k)
                {
                    ix = abs(off_u + k * oversample);
                    p = 2 * (kernel_start + iy * conv_size_half + ix);
                    cwt[0] = conv_func[p];
                    cwt[1] = -conv_func[p + 1]; /* Conjugate. */
                    sum += cwt[0]; /* Real part only. */
                    p = 2 * (((grid_v + j) * grid_size) + grid_u + k);
                    grid[p]     += (val[0] * cwt[0] - val[1] * cwt[1]);
                    grid[p + 1] += (val[1] * cwt[0] + val[0] * cwt[1]);
                }
            }
        }
        else
        {
            for (j = -wsupport; j <= wsupport; ++j)
            {
                iy = abs(off_v + j * oversample);
                for (k = -wsupport; k <= wsupport; ++k)
                {
                    ix = abs(off_u + k * oversample);
                    p = 2 * (kernel_start + iy * conv_size_half + ix);
                    cwt[0] = conv_func[p];
                    cwt[1] = conv_func[p + 1];
                    sum += cwt[0]; /* Real part only. */
                    p = 2 * (((grid_v + j) * grid_size) + grid_u + k);
                    grid[p]     += (val[0] * cwt[0] - val[1] * cwt[1]);
                    grid[p + 1] += (val[1] * cwt[0] + val[0] * cwt[1]);
                }
            }
        }
        *norm += sum * w;
    }
}


#ifdef __cplusplus
}
#endif
