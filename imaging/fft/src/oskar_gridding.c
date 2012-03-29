/*
 * Copyright (c) 2011, The University of Oxford
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

#include "imaging/fft/oskar_gridding.h"

#include "utility/oskar_vector_types.h"

#include "math.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_offset(double x, double pixel_size,
        unsigned kernel_oversample, int* grid_idx, int* kernel_idx);
double oskar_round_away_from_zero(double x);
double oskar_round_towards_zero(double x);



double oskar_grid_standard(const oskar_Visibilities* vis,
        const oskar_GridKernel_d* kernel, oskar_VisGrid_d* grid)
{
    /*int i, iy, ix;*/
    double grid_sum;
    /*int support, g_centre;*/
    /*int ix_grid, iy_grid, ix_kernel, iy_kernel;*/

    /*support  = (kernel->num_cells - 1) / 2;*/
    /* fixme: Will only work for even grids, should use floor() ...? */
    /*g_centre = grid->size / 2.0;*/

    grid_sum = 0.0;

    /*
    for (i = 0; i < vis->num_samples; ++i)
    {
        oskar_evaluate_offset(vis->u[i], grid->pixel_separation,
                (unsigned)kernel->oversample, &ix_grid, &ix_kernel);
        oskar_evaluate_offset(vis->v[i], grid->pixel_separation,
                (unsigned)kernel->oversample, &iy_grid, &iy_kernel);

        const double vis_re = vis->amp[i].x;
        const double vis_im = vis->amp[i].y;

        ix_grid   += g_centre;
        iy_grid   += g_centre;
        ix_kernel += kernel->centre;
        iy_kernel += kernel->centre;

        for (iy = -support; iy <= support; ++iy)
        {
            const int gy = iy + iy_grid;
            if (gy >= grid->size) continue;

            for (ix = -support; ix <= support; ++ix)
            {
                const int gx = ix + ix_grid;
                if (gx >= grid->size) continue;

                const int g_idx = gy * grid->size + gx;

                const int kx = (ix * kernel->oversample) + ix_kernel;
                const int ky = (iy * kernel->oversample) + iy_kernel;
                const double kamp = kernel->amp[kx] * kernel->amp[ky];

                grid->amp[g_idx].x += kamp * vis_re;
                grid->amp[g_idx].y += kamp * vis_im;
                grid_sum += kamp;
            }
        }
    }
    */

    return grid_sum;
}


void oskar_evaluate_offset(double x, double pixel_size,
        unsigned kernel_oversample, int* grid_idx, int* kernel_idx)
{
    double x_scaled, grid_delta, kernel_delta;

    /* Scale input coordinate to grid space units. */
    x_scaled = x / pixel_size;

    /* Evaluate the closest grid cell. */
    *grid_idx = (int) oskar_round_towards_zero(x_scaled);

    /* Evaluate the index of the convolution kernel at the closest grid cell. */
    grid_delta = (*grid_idx - x_scaled);
    kernel_delta = grid_delta * kernel_oversample;
    *kernel_idx = (int) oskar_round_towards_zero(kernel_delta);
}


double oskar_round_away_from_zero(double x)
{
    return (x > 0.0f) ? floor(x + 0.5) : ceil(x - 0.5);
}


double oskar_round_towards_zero(double x)
{
    return (x > 0.0f) ? ceil(x - 0.5) : floor(x + 0.5);
}

#ifdef __cplusplus
}
#endif

