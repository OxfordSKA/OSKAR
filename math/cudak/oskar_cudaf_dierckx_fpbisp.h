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

#ifndef OSKAR_CUDAF_DIERCKX_FPBISP_H_
#define OSKAR_CUDAF_DIERCKX_FPBISP_H_

/**
 * @file oskar_cudaf_dierckx_fpbisp.h
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
__device__ void oskar_cudaf_dierckx_fpbisp_f(const float *tx, const int nx,
        const float *ty, const int ny, const float *c, const int kx,
        const int ky, const float *x, const int mx, const float *y,
        const int my, float *z, float *wx, float *wy, int *lx, int *ly)
{
    /* Local variables */
    int i, j, l, m, i1, j1, l1, l2, kx1, ky1, nkx1, nky1;
    float h[6];
    float tb, te, sp, arg;

    /* Parameter adjustments */
    --tx;
    --ty;
    --c;
    --lx;
    wx -= (1 + mx);
    --x;
    --ly;
    wy -= (1 + my);
    --z;
    --y;

    /* Function Body */
    kx1 = kx + 1;
    nkx1 = nx - kx1;
    tb = tx[kx1];
    te = tx[nkx1 + 1];
    l = kx1;
    for (i = 1; i <= mx; ++i)
    {
        arg = x[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < tx[l + 1] || l == nkx1)) l++;
        oskar_cudaf_dierckx_fpbspl_f(&tx[1], kx, arg, l, h);
        lx[i] = l - kx1;
        for (j = 1; j <= kx1; ++j)
        {
            wx[i + j * mx] = h[j - 1];
        }
    }
    ky1 = ky + 1;
    nky1 = ny - ky1;
    tb = ty[ky1];
    te = ty[nky1 + 1];
    l = ky1;
    for (i = 1; i <= my; ++i)
    {
        arg = y[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < ty[l + 1] || l == nky1)) l++;
        oskar_cudaf_dierckx_fpbspl_f(&ty[1], ky, arg, l, h);
        ly[i] = l - ky1;
        for (j = 1; j <= ky1; ++j)
        {
            wy[i + j * my] = h[j - 1];
        }
    }
    m = 0;
    for (i = 1; i <= mx; ++i)
    {
        l = lx[i] * nky1;
        for (i1 = 1; i1 <= kx1; ++i1)
        {
            h[i1 - 1] = wx[i + i1 * mx];
        }
        for (j = 1; j <= my; ++j)
        {
            l1 = l + ly[j];
            sp = 0.0f;
            for (i1 = 1; i1 <= kx1; ++i1)
            {
                l2 = l1;
                for (j1 = 1; j1 <= ky1; ++j1)
                {
                    ++l2;
                    sp += c[l2] * h[i1 - 1] * wy[j + j1 * my];
                }
                l1 += nky1;
            }
            ++m;
            z[m] = sp;
        }
    }
}

/**
 * @brief
 * CUDA device function for fpbisp from DIERCKX library (double precision).
 *
 * @details
 * CUDA device function to replace the fpbisp function from the DIERCKX
 * fitting library.
 */
__device__ void oskar_cudaf_dierckx_fpbisp_d(const double *tx, const int nx,
        const double *ty, const int ny, const double *c, const int kx,
        const int ky, const double *x, const int mx, const double *y,
        const int my, double *z, double *wx, double *wy, int *lx, int *ly)
{
    /* Local variables */
    int i, j, l, m, i1, j1, l1, l2, kx1, ky1, nkx1, nky1;
    double h[6];
    double tb, te, sp, arg;

    /* Parameter adjustments */
    --tx;
    --ty;
    --c;
    --lx;
    wx -= (1 + mx);
    --x;
    --ly;
    wy -= (1 + my);
    --z;
    --y;

    /* Function Body */
    kx1 = kx + 1;
    nkx1 = nx - kx1;
    tb = tx[kx1];
    te = tx[nkx1 + 1];
    l = kx1;
    for (i = 1; i <= mx; ++i)
    {
        arg = x[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < tx[l + 1] || l == nkx1)) l++;
        oskar_cudaf_dierckx_fpbspl_d(&tx[1], kx, arg, l, h);
        lx[i] = l - kx1;
        for (j = 1; j <= kx1; ++j)
        {
            wx[i + j * mx] = h[j - 1];
        }
    }
    ky1 = ky + 1;
    nky1 = ny - ky1;
    tb = ty[ky1];
    te = ty[nky1 + 1];
    l = ky1;
    for (i = 1; i <= my; ++i)
    {
        arg = y[i];
        if (arg < tb) arg = tb;
        if (arg > te) arg = te;
        while (!(arg < ty[l + 1] || l == nky1)) l++;
        oskar_cudaf_dierckx_fpbspl_d(&ty[1], ky, arg, l, h);
        ly[i] = l - ky1;
        for (j = 1; j <= ky1; ++j)
        {
            wy[i + j * my] = h[j - 1];
        }
    }
    m = 0;
    for (i = 1; i <= mx; ++i)
    {
        l = lx[i] * nky1;
        for (i1 = 1; i1 <= kx1; ++i1)
        {
            h[i1 - 1] = wx[i + i1 * mx];
        }
        for (j = 1; j <= my; ++j)
        {
            l1 = l + ly[j];
            sp = 0.0;
            for (i1 = 1; i1 <= kx1; ++i1)
            {
                l2 = l1;
                for (j1 = 1; j1 <= ky1; ++j1)
                {
                    ++l2;
                    sp += c[l2] * h[i1 - 1] * wy[j + j1 * my];
                }
                l1 += nky1;
            }
            ++m;
            z[m] = sp;
        }
    }
}

#endif // OSKAR_CUDAF_DIERCKX_FPBISP_H_
