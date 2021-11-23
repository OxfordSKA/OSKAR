/*
 * Copyright (c) 2018-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_grid_wproj2.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_wproj2_d(
        const size_t num_w_planes,
        const int* RESTRICT support,
        const int oversample,
        const int* wkernel_start,
        const double* RESTRICT wkernel,
        const size_t num_points,
        const double* RESTRICT uu,
        const double* RESTRICT vv,
        const double* RESTRICT ww,
        const double* RESTRICT vis,
        const double* RESTRICT weight,
        const double cell_size_rad,
        const double w_scale,
        const int grid_size,
        size_t* RESTRICT num_skipped,
        double* RESTRICT norm,
        double* RESTRICT grid)
{
    size_t i = 0;
    const int grid_centre = grid_size / 2;
    const int oversample_h = oversample / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j = 0, k = 0;

        /* Convert UV coordinates to grid coordinates. */
        const double pos_u = -uu[i] * grid_scale;
        const double pos_v = vv[i] * grid_scale;
        const double ww_i = ww[i];
        const double conv_conj = (ww_i > 0.0) ? -1.0 : 1.0;
        const size_t grid_w = (size_t)round(sqrt(fabs(ww_i * w_scale)));
        const int grid_u = (int)round(pos_u) + grid_centre;
        const int grid_v = (int)round(pos_v) + grid_centre;

        /* Get visibility data. */
        const double weight_i = weight[i];
        const double v_re = weight_i * vis[2 * i];
        const double v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)round((round(pos_u) - pos_u) * oversample);
        const int off_v = (int)round((round(pos_v) - pos_v) * oversample);

        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];
        const int kernel_start = grid_w < num_w_planes ?
                wkernel_start[grid_w] : wkernel_start[num_w_planes - 1];

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        const int conv_len = 2 * w_support + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        const int mid = kernel_start + (abs(off_u) + 1) * width - 1 - w_support;
        const int stride = (off_u >= 0) ? 1 : -1;
        for (j = -w_support; j <= w_support; ++j)
        {
            const int t = mid - abs(off_v + j * oversample) * conv_len;
            size_t p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -w_support; k <= w_support; ++k)
            {
                const int p = (t + stride * k) << 1;
                const double c_re = wkernel[p];
                const double c_im = wkernel[p + 1] * conv_conj;
                const size_t p2 = (p1 + k) << 1;
                grid[p2]     += (v_re * c_re - v_im * c_im);
                grid[p2 + 1] += (v_im * c_re + v_re * c_im);
                sum += c_re; /* Real part only. */
            }
        }
        *norm += sum * weight_i;
    }
}


void oskar_grid_wproj2_f(
        const size_t num_w_planes,
        const int* RESTRICT support,
        const int oversample,
        const int* wkernel_start,
        const float* RESTRICT wkernel,
        const size_t num_points,
        const float* RESTRICT uu,
        const float* RESTRICT vv,
        const float* RESTRICT ww,
        const float* RESTRICT vis,
        const float* RESTRICT weight,
        const float cell_size_rad,
        const float w_scale,
        const int grid_size,
        size_t* RESTRICT num_skipped,
        double* RESTRICT norm,
        float* RESTRICT grid)
{
    size_t i = 0;
    const int grid_centre = grid_size / 2;
    const int oversample_h = oversample / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j = 0, k = 0;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const float ww_i = ww[i];
        const float conv_conj = (ww_i > 0.0f) ? -1.0f : 1.0f;
        const size_t grid_w = (size_t)roundf(sqrtf(fabsf(ww_i * w_scale)));
        const int grid_u = (int)roundf(pos_u) + grid_centre;
        const int grid_v = (int)roundf(pos_v) + grid_centre;

        /* Get visibility data. */
        const float weight_i = weight[i];
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)roundf((roundf(pos_u) - pos_u) * oversample);
        const int off_v = (int)roundf((roundf(pos_v) - pos_v) * oversample);

        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];
        const int kernel_start = grid_w < num_w_planes ?
                wkernel_start[grid_w] : wkernel_start[num_w_planes - 1];

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        const int conv_len = 2 * w_support + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        const int mid = kernel_start + (abs(off_u) + 1) * width - 1 - w_support;
        const int stride = (off_u >= 0) ? 1 : -1;
        for (j = -w_support; j <= w_support; ++j)
        {
            const int t = mid - abs(off_v + j * oversample) * conv_len;
            size_t p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            for (k = -w_support; k <= w_support; ++k)
            {
                const int p = (t + stride * k) << 1;
                const float c_re = wkernel[p];
                const float c_im = wkernel[p + 1] * conv_conj;
                const size_t p2 = (p1 + k) << 1;
                grid[p2]     += (v_re * c_re - v_im * c_im);
                grid[p2 + 1] += (v_im * c_re + v_re * c_im);
                sum += c_re; /* Real part only. */
            }
        }
        *norm += sum * weight_i;
    }
}

#ifdef __cplusplus
}
#endif
