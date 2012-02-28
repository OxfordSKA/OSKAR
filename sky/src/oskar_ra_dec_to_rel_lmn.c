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

#include "sky/oskar_ra_dec_to_rel_lmn.h"
#include "math/oskar_sph_to_lm.h"
#include "sky/oskar_lm_to_n.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
int oskar_ra_dec_to_rel_lmn_f(int n, const float* h_ra, const float* h_dec,
        float ra0, float dec0, float* h_l, float* h_m, float* h_n)
{
    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    oskar_sph_to_lm_f(n, ra0, dec0, h_ra, h_dec, h_l, h_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_lm_to_n_f(n, h_l, h_m, h_n);
    return 0;
}

/* Double precision. */
int oskar_ra_dec_to_rel_lmn_d(int n, const double* h_ra, const double* h_dec,
        double ra0, double dec0, double* h_l, double* h_m, double* h_n)
{
    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    oskar_sph_to_lm_d(n, ra0, dec0, h_ra, h_dec, h_l, h_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_lm_to_n_d(n, h_l, h_m, h_n);
    return 0;
}

#ifdef __cplusplus
}
#endif
