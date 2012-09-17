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

#include "sky/oskar_sky_model_resize.h"
#include "utility/oskar_mem_realloc.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_model_resize(oskar_SkyModel* sky, int num_sources, int* status)
{
    /* Check all inputs. */
    if (!sky || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    sky->num_sources = num_sources;

    /* Resize the model data. */
    oskar_mem_realloc(&sky->RA, num_sources, status);
    oskar_mem_realloc(&sky->Dec, num_sources, status);
    oskar_mem_realloc(&sky->I, num_sources, status);
    oskar_mem_realloc(&sky->Q, num_sources, status);
    oskar_mem_realloc(&sky->U, num_sources, status);
    oskar_mem_realloc(&sky->V, num_sources, status);
    oskar_mem_realloc(&sky->reference_freq, num_sources, status);
    oskar_mem_realloc(&sky->spectral_index, num_sources, status);
    oskar_mem_realloc(&sky->rel_l, num_sources, status);
    oskar_mem_realloc(&sky->rel_m, num_sources, status);
    oskar_mem_realloc(&sky->rel_n, num_sources, status);
    oskar_mem_realloc(&sky->FWHM_major, num_sources, status);
    oskar_mem_realloc(&sky->FWHM_minor, num_sources, status);
    oskar_mem_realloc(&sky->position_angle, num_sources, status);
    oskar_mem_realloc(&sky->gaussian_a, num_sources, status);
    oskar_mem_realloc(&sky->gaussian_b, num_sources, status);
    oskar_mem_realloc(&sky->gaussian_c, num_sources, status);
}

#ifdef __cplusplus
}
#endif
