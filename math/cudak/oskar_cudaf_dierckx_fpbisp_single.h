/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_CUDAF_DIERCKX_FPBISP_SINGLE_H_
#define OSKAR_CUDAF_DIERCKX_FPBISP_SINGLE_H_

/**
 * @file oskar_cudaf_dierckx_fpbisp_single.h
 */

#include "oskar_global.h"
#include "math/cudak/oskar_cudaf_dierckx_fpbspl.h"

/**
 * @brief
 * CUDA device function for fpbisp from DIERCKX library (single precision).
 *
 * @details
 * CUDA device function to replace the fpbisp function from the DIERCKX
 * fitting library.
 */
__device__
void oskar_cudaf_dierckx_fpbisp_single_f(const float *tx, const int nx,
        const float *ty, const int ny, const float *c, const int kx,
        const int ky, float x, float y, float *z)
{
    int j, l, l1, l2, k1, nk1, lx;
    float wx[6], wy[6], t;

    // Do x.
    k1 = kx + 1;
    nk1 = nx - k1;
    t = tx[kx];
    if (x < t) x = t;
    t = tx[nk1];
    if (x > t) x = t;
    l = k1;
    while (!(x < tx[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_f(tx, kx, x, l, wx);
    lx = l - k1;

    // Do y.
    k1 = ky + 1;
    nk1 = ny - k1;
    t = ty[ky];
    if (y < t) y = t;
    t = ty[nk1];
    if (y > t) y = t;
    l = k1;
    while (!(y < ty[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_f(ty, ky, y, l, wy);
    l1 = lx * nk1 + (l - k1);

    // Evaluate surface using coefficients.
    t = 0.0f;
    for (l = 0; l <= kx; ++l)
    {
        l2 = l1;
        for (j = 0; j <= ky; ++j)
        {
            t += c[l2] * wx[l] * wy[j];
            ++l2;
        }
        l1 += nk1;
    }
    *z = t;
}

/**
 * @brief
 * CUDA device function for fpbisp from DIERCKX library (double precision).
 *
 * @details
 * CUDA device function to replace the fpbisp function from the DIERCKX
 * fitting library.
 */
__device__
void oskar_cudaf_dierckx_fpbisp_single_d(const double *tx, const int nx,
        const double *ty, const int ny, const double *c, const int kx,
        const int ky, double x, double y, double *z)
{
    int j, l, l1, l2, k1, nk1, lx;
    double wx[6], wy[6], t;

    // Do x.
    k1 = kx + 1;
    nk1 = nx - k1;
    t = tx[kx];
    if (x < t) x = t;
    t = tx[nk1];
    if (x > t) x = t;
    l = k1;
    while (!(x < tx[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_d(tx, kx, x, l, wx);
    lx = l - k1;

    // Do y.
    k1 = ky + 1;
    nk1 = ny - k1;
    t = ty[ky];
    if (y < t) y = t;
    t = ty[nk1];
    if (y > t) y = t;
    l = k1;
    while (!(y < ty[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_d(ty, ky, y, l, wy);
    l1 = lx * nk1 + (l - k1);

    // Evaluate surface using coefficients.
    t = 0.0;
    for (l = 0; l <= kx; ++l)
    {
        l2 = l1;
        for (j = 0; j <= ky; ++j)
        {
            t += c[l2] * wx[l] * wy[j];
            ++l2;
        }
        l1 += nk1;
    }
    *z = t;
}

#endif // OSKAR_CUDAF_DIERCKX_FPBISP_SINGLE_H_
