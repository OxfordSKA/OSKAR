/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/oskar_grid_weights.h"
#include "math/oskar_kahan_sum.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_weights_write_d(const size_t num_points,
        const double* RESTRICT uu, const double* RESTRICT vv,
        const double* RESTRICT weight, const double cell_size_rad,
        const int grid_size, size_t* RESTRICT num_skipped,
        double* RESTRICT grid)
{
    size_t i = 0;
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
        const double* RESTRICT uu, const double* RESTRICT vv,
        const double* RESTRICT weight_in, double* RESTRICT weight_out,
        const double cell_size_rad, const int grid_size,
        size_t* RESTRICT num_skipped, const double* RESTRICT grid)
{
    size_t i = 0;
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
        const float* RESTRICT uu, const float* RESTRICT vv,
        const float* RESTRICT weight, const float cell_size_rad,
        const int grid_size, size_t* RESTRICT num_skipped,
        float* RESTRICT grid, float* RESTRICT grid_guard)
{
    size_t i = 0;
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

        /* Add weight to the grid using Kahan summation. */
        OSKAR_KAHAN_SUM(float, grid[t], weight[i], grid_guard[t]);
    }
}

void oskar_grid_weights_read_f(const size_t num_points,
        const float* RESTRICT uu, const float* RESTRICT vv,
        const float* RESTRICT weight_in, float* RESTRICT weight_out,
        const float cell_size_rad, const int grid_size,
        size_t* RESTRICT num_skipped, const float* RESTRICT grid)
{
    size_t i = 0;
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
