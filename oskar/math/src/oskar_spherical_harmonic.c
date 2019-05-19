/*
 * Copyright (c) 2019, The University of Oxford
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

#include "math/oskar_spherical_harmonic.h"
#include "math/private_spherical_harmonic.h"

#ifdef __cplusplus
extern "C" {
#endif

const oskar_Mem* oskar_spherical_harmonic_coeff_const(
        const oskar_SphericalHarmonic* h)
{
    return h->coeff;
}

void oskar_spherical_harmonic_copy(oskar_SphericalHarmonic* dst,
        const oskar_SphericalHarmonic* src, int* status)
{
    dst->l_max = src->l_max;
    if (src->l_max >= 0)
        oskar_mem_copy(dst->coeff, src->coeff, status);
}

oskar_SphericalHarmonic* oskar_spherical_harmonic_create(int type,
        int location, int l_max, int* status)
{
    oskar_SphericalHarmonic* h = (oskar_SphericalHarmonic*)
            calloc(1, sizeof(oskar_SphericalHarmonic));
    h->l_max = l_max;
    h->coeff = oskar_mem_create(type, location, 0, status);
    if (l_max >= 0)
    {
        const int num_coeff = (l_max + 1) * (l_max + 1);
        oskar_mem_realloc(h->coeff, num_coeff, status);
    }
    return h;
}

void oskar_spherical_harmonic_free(oskar_SphericalHarmonic* h)
{
    int status = 0;
    if (!h) return;
    oskar_mem_free(h->coeff, &status);
    free(h);
}

int oskar_spherical_harmonic_l_max(const oskar_SphericalHarmonic* h)
{
    return h->l_max;
}

void oskar_spherical_harmonic_set_coeff(oskar_SphericalHarmonic* h,
        int l_max, const oskar_Mem* coeff, int* status)
{
    const int num_coeff = (l_max + 1) * (l_max + 1);
    if (oskar_mem_length(coeff) < (size_t) num_coeff)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    h->l_max = l_max;
    oskar_mem_copy(h->coeff, coeff, status);
}

#ifdef __cplusplus
}
#endif
