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

#include "imager/oskar_grid_weights.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_weights_write_d(const size_t num_points,
        const double* restrict uu, const double* restrict vv,
        const double* restrict weight, const double cell_size_rad,
        const int grid_size, size_t* restrict num_skipped,
        double* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Grid the existing weights. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        /* Convert UV coordinates to grid coordinates. */
        const int grid_u = (int)round(-uu[i] * grid_scale) + grid_centre;
        const int grid_v = (int)round(vv[i] * grid_scale) + grid_centre;
        size_t t = grid_v;
        t *= grid_size; /* Tested to avoid int overflow. */
        t += grid_u;

        /* Catch points that would lie outside the grid. */
        if (grid_u >= grid_size || grid_u < 0 ||
                grid_v >= grid_size || grid_v < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Add weight to the grid. */
        grid[t] += weight[i];
    }
}

void oskar_grid_weights_read_d(const size_t num_points,
        const double* restrict uu, const double* restrict vv,
        const double* restrict weight_in, double* restrict weight_out,
        const double cell_size_rad, const int grid_size,
        size_t* restrict num_skipped, const double* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Look up gridded weight density at each point location. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        /* Convert UV coordinates to grid coordinates. */
        const int grid_u = (int)round(-uu[i] * grid_scale) + grid_centre;
        const int grid_v = (int)round(vv[i] * grid_scale) + grid_centre;
        size_t t = grid_v;
        t *= grid_size; /* Tested to avoid int overflow. */
        t += grid_u;

        /* Catch points that would lie outside the grid. */
        if (grid_u >= grid_size || grid_u < 0 ||
                grid_v >= grid_size || grid_v < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Calculate new weight based on gridded point density. */
        weight_out[i] = (grid[t] != 0.0) ? weight_in[i] / grid[t] : 0.0;
    }
}

void oskar_grid_weights_write_f(const size_t num_points,
        const float* restrict uu, const float* restrict vv,
        const float* restrict weight, const float cell_size_rad,
        const int grid_size, size_t* restrict num_skipped,
        float* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Grid the existing weights. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        /* Convert UV coordinates to grid coordinates. */
        const int grid_u = (int)roundf(-uu[i] * grid_scale) + grid_centre;
        const int grid_v = (int)roundf(vv[i] * grid_scale) + grid_centre;
        size_t t = grid_v;
        t *= grid_size; /* Tested to avoid int overflow. */
        t += grid_u;

        /* Catch points that would lie outside the grid. */
        if (grid_u >= grid_size || grid_u < 0 ||
                grid_v >= grid_size || grid_v < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Add weight to the grid. */
        grid[t] += weight[i];
    }
}

void oskar_grid_weights_read_f(const size_t num_points,
        const float* restrict uu, const float* restrict vv,
        const float* restrict weight_in, float* restrict weight_out,
        const float cell_size_rad, const int grid_size,
        size_t* restrict num_skipped, const float* restrict grid)
{
    size_t i;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Look up gridded weight density at each point location. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        /* Convert UV coordinates to grid coordinates. */
        const int grid_u = (int)roundf(-uu[i] * grid_scale) + grid_centre;
        const int grid_v = (int)roundf(vv[i] * grid_scale) + grid_centre;
        size_t t = grid_v;
        t *= grid_size; /* Tested to avoid int overflow. */
        t += grid_u;

        /* Catch points that would lie outside the grid. */
        if (grid_u >= grid_size || grid_u < 0 ||
                grid_v >= grid_size || grid_v < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Calculate new weight based on gridded point density. */
        weight_out[i] = (grid[t] != 0.0) ? weight_in[i] / grid[t] : 0.0;
    }
}

#ifdef __cplusplus
}
#endif
