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

#include "imaging/oskar_evaluate_image_lm_grid.h"
#include "math/oskar_linspace.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_image_lm_grid_f(int num_l, int num_m,
        float fov_lon, float fov_lat, float* grid_l, float* grid_m)
{
    int i, j, p;
    float l_max, m_max, *l, *m;

    /* Allocate temporary memory for vectors. */
    l = malloc(num_l * sizeof(float));
    m = malloc(num_m * sizeof(float));

    /* Set up the grid boundaries. */
    l_max = sin(0.5 * fov_lon);
    m_max = sin(0.5 * fov_lat);

    /* Create the axis vectors. */
    oskar_linspace_f(l, l_max, -l_max, num_l); /* FITS convention. */
    oskar_linspace_f(m, -m_max, m_max, num_m);

    /* Slowest varying is m, fastest varying is l. */
    for (j = 0, p = 0; j < num_m; ++j)
    {
        for (i = 0; i < num_l; ++i, ++p)
        {
            grid_l[p] = l[i];
            grid_m[p] = m[j];
        }
    }

    /* Free temporary memory. */
    free(l);
    free(m);
}

/* Double precision. */
void oskar_evaluate_image_lm_grid_d(int num_l, int num_m,
        double fov_lon, double fov_lat, double* grid_l, double* grid_m)
{
    int i, j, p;
    double l_max, m_max, *l, *m;

    /* Allocate temporary memory for vectors. */
    l = malloc(num_l * sizeof(double));
    m = malloc(num_m * sizeof(double));

    /* Set up the grid boundaries. */
    l_max = sin(0.5 * fov_lon);
    m_max = sin(0.5 * fov_lat);

    /* Create the axis vectors. */
    oskar_linspace_d(l, l_max, -l_max, num_l); /* FITS convention. */
    oskar_linspace_d(m, -m_max, m_max, num_m);

    /* Slowest varying is m, fastest varying is l. */
    for (j = 0, p = 0; j < num_m; ++j)
    {
        for (i = 0; i < num_l; ++i, ++p)
        {
            grid_l[p] = l[i];
            grid_m[p] = m[j];
        }
    }

    /* Free temporary memory. */
    free(l);
    free(m);
}

#ifdef __cplusplus
}
#endif
