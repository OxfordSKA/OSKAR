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


#include "sky/oskar_sky_model_copy.h"
#include "utility/oskar_mem_copy.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_copy(oskar_SkyModel* dst, const oskar_SkyModel* src)
{
    int error = OSKAR_SUCCESS;

    if (src == NULL || dst == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Copy the memory blocks */
    error = oskar_mem_copy(&dst->RA, &src->RA);
    if (error) return error;
    error = oskar_mem_copy(&dst->Dec, &src->Dec);
    if (error) return error;
    error = oskar_mem_copy(&dst->I, &src->I);
    if (error) return error;
    error = oskar_mem_copy(&dst->Q, &src->Q);
    if (error) return error;
    error = oskar_mem_copy(&dst->U, &src->U);
    if (error) return error;
    error = oskar_mem_copy(&dst->V, &src->V);
    if (error) return error;
    error = oskar_mem_copy(&dst->reference_freq, &src->reference_freq);
    if (error) return error;
    error = oskar_mem_copy(&dst->spectral_index, &src->spectral_index);
    if (error) return error;
    error = oskar_mem_copy(&dst->rel_l, &src->rel_l);
    if (error) return error;
    error = oskar_mem_copy(&dst->rel_m, &src->rel_m);
    if (error) return error;
    error = oskar_mem_copy(&dst->rel_n, &src->rel_n);
    if (error) return error;

    /* Copy meta data */
    dst->num_sources = src->num_sources;

    return error;
}


#ifdef __cplusplus
}
#endif
