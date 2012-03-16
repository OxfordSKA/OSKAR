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

#include "imaging/fft/oskar_grid_correct.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_grid_correction_d(oskar_GridKernel_d* kernel,
        const unsigned grid_size, double** correction)
{
	int i, j, grid_centre;
	double *kernel_x, *f_sinc, *c_sinc, *correction1d;
	double grid_inc, f, x, absx, correction_max;


    /* === Kernel x grid coordinates. */
    kernel_x = (double*) malloc(kernel->size * sizeof(double));
    for (i = 0; i < kernel->size; ++i)
        kernel_x[i] = (i - kernel->centre) * kernel->xinc;

    /* === Sinc term sinc(pi * f * x) / (pi * f * x) */
    grid_centre = grid_size / 2;
    grid_inc = 1.0 / grid_size;
    f_sinc = (double*) malloc(grid_size * sizeof(double));
    f = kernel->xinc;
    for (i = 0; i < (int)grid_size; ++i)
    {
        x    = (i - grid_centre) * grid_inc;
        absx = fabs(x);
        if (absx < DBL_EPSILON)
            f_sinc[i] = 1.0;
        else
            f_sinc[i] = sin(M_PI * f * absx) / (M_PI * f * absx);
    }

    /* === DFT of convolution function */
    c_sinc = (double*) malloc(grid_size * sizeof(double));
    memset(c_sinc, 0, grid_size * sizeof(double));
    for (j = 0; j < (int)grid_size; ++j)
    {
        x = (j - grid_centre) * grid_inc;
        for (i = 0; i < kernel->size; ++i)
        {
            c_sinc[j] += kernel->amp[i] * cos(-2 * M_PI * x * kernel_x[i]);
        }
    }

    /* === Combine terms to form 1d correction function. */
    correction1d = (double*) malloc(grid_size * sizeof(double));
    for (i = 0; i < (int)grid_size; ++i)
    {
        correction1d[i] = f_sinc[i] * c_sinc[i];
        /* correction1d[i] = f_sinc[i]; */
        /* correction1d[i] = c_sinc[i]; */
    }

    /* === Find the maximum of the correction function. */
    /* Note: same as normalising to the centre? */
    correction_max = -DBL_MAX;
    for (i = 0; i < (int)grid_size; ++i)
        correction_max = MAX(correction_max, correction1d[i]);

    /* === Normalise to the maximum. */
    for (i = 0; i < (int)grid_size; ++i)
        correction1d[i] /= correction_max;

    /* === Convert to a 2D correction screen. */
    *correction = (double*) malloc(grid_size * grid_size * sizeof(double));
    for (j = 0; j < (int)grid_size; ++j)
    {
        for (i = 0; i < (int)grid_size; ++i)
        {
            (*correction)[j * grid_size + i] = correction1d[i] * correction1d[j];
        }
    }

    /* == Clean up. */
    free(kernel_x);
    free(f_sinc);
    free(c_sinc);
    free(correction1d);
}

#ifdef __cplusplus
}
#endif

