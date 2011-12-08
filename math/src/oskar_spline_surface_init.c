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

#include "math/oskar_spline_surface_init.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fortran function prototype. */
void regrid_(int* iopt, int* mx, const float x[], int* my, const float y[],
        const float z[], const float* xb, const float* xe, const float* yb,
        const float* ye, int* kx, int* ky, float* s, const int* nxest,
        const int* nyest, int* nx, float tx[], int* ny, float ty[], float c[],
        float* fp, float wrk[], int* lwrk, int iwrk[], int* kwrk, int* ier);

int oskar_spline_surface_init_f(int num_x, const float* x, int num_y,
        const float* y, const float* data, int kx, float* tx,
        int ky, float* ty, int* nx, int* ny, float* c)
{
    int k, iopt, fail, ier, fix_x, fix_y, u, lwrk, kwrk, nxest, nyest;
    float noise, threshold, s, fp;

    /* Work arrays. */
    float* wrk;
    int* iwrk;

    /* Set options and limits. */
    iopt = 0;
    noise = 10e-4;
    threshold = pow(2.0 * noise, 2.0);
    fix_x = 2 * (kx + 1);
    fix_y = 2 * (ky + 1);
    nxest = num_x + kx + 1;
    nyest = num_y + ky + 1;

    /* Compute sizes of work arrays. */
    u = num_y > nxest ? num_y : nxest;
    lwrk = 4 + nxest * (num_y + 2 * kx + 5) +
            nyest * (2 * ky + 5) + num_x * (kx + 1) +
            num_y * (ky + 1) + u;
    kwrk = 3 + num_x + num_y + nxest + nyest;

    /* Initialise work arrays. */
    wrk  = (float*) malloc(lwrk * sizeof(float));
    iwrk = (int*)   malloc(kwrk * sizeof(int));

    /* Set initial smoothing factor (ignored for iopt < 0). */
    s = num_x * num_y * (noise * noise);

    /* Compute splines. */
    k = 0;
    fail = 0;
    do
    {
        /* Generate knot positions *at grid points* if required. */
        if (iopt < 0 || fail)
        {
            int i, ki, stride = 1;
            for (ki = 0, i = kx - 1; i <= num_x - kx + stride; i += stride, ++ki)
                tx[ki + kx + 1] = x[i]; /* Knot x positions. */
            *nx = ki + 2 * kx + 1;
            for (ki = 0, i = ky - 1; i <= num_y - ky + stride; i += stride, ++ki)
                ty[ki + ky + 1] = y[i]; /* Knot y positions. */
            *ny = ki + 2 * ky + 1;
        }

        /* Set iopt to 1 if this is at least the second of multiple passes. */
        if (k > 0) iopt = 1;
        regrid_(&iopt, &num_x, x, &num_y, y, data, &x[0], &x[num_x-1], &y[0],
                &y[num_y-1], &kx, &ky, &s, &nxest, &nyest, nx, tx, ny, ty,
                c, &fp, wrk, &lwrk, iwrk, &kwrk, &ier);
        if (ier == 1)
        {
            fprintf(stderr, "ERROR: Workspace overflow.\n");
            break;
        }
        else if (ier == 2)
        {
            fprintf(stderr, "ERROR: Impossible result! (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            ier = 0;
            fail = 1;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 3)
        {
            fprintf(stderr, "ERROR: Iteration limit. (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            ier = 0;
            fail = 1;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 10)
        {
            fprintf(stderr, "ERROR: Invalid input arguments.\n");
            break;
        }
        fail = 0;

        /* Reduce smoothness parameter. */
        s = s / 1.2;

        /* Increment counter. */
        ++k;
    } while (fp / ((*nx - fix_x) * (*ny - fix_y)) > threshold && k < 1000
            && (iopt >= 0 || fail));

    /* Free memory and return. */
    free(iwrk);
    free(wrk);
    if (ier != 0) return -1000 - ier;
    return 0;
}

#ifdef __cplusplus
}
#endif
