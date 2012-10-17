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

#include "sky/oskar_ra_dec_to_rel_lmn_cuda.h"
#include "math/cudak/oskar_cudak_sph_to_lm.h"
#include "sky/cudak/oskar_cudak_lm_to_n.h"

/* Single precision. */
void oskar_ra_dec_to_rel_lmn_cuda_f(int num_points, const float* d_ra,
        const float* d_dec, float ra0, float dec0, float* d_l, float* d_m,
        float* d_n)
{
    float cosDec0, sinDec0;
    int num_blocks, num_threads = 256;

    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cosDec0 = cosf(dec0);
    sinDec0 = sinf(dec0);
    oskar_cudak_sph_to_lm_f OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_points, d_ra, d_dec, ra0, cosDec0, sinDec0, d_l, d_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_cudak_lm_to_n_f OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points,
            d_l, d_m, d_n);
}

/* Double precision. */
void oskar_ra_dec_to_rel_lmn_cuda_d(int num_points, const double* d_ra,
        const double* d_dec, double ra0, double dec0, double* d_l, double* d_m,
        double* d_n)
{
    double cosDec0, sinDec0;
    int num_blocks, num_threads = 256;

    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cosDec0 = cos(dec0);
    sinDec0 = sin(dec0);
    oskar_cudak_sph_to_lm_d OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_points, d_ra, d_dec, ra0, cosDec0, sinDec0, d_l, d_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_cudak_lm_to_n_d OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points,
            d_l, d_m, d_n);
}
