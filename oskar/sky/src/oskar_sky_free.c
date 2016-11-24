/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"
#include "mem/oskar_mem.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_free(oskar_Sky* model, int* status)
{
    if (!model) return;

    /* Free the memory. */
    oskar_mem_free(model->ra_rad, status);
    oskar_mem_free(model->dec_rad, status);
    oskar_mem_free(model->I, status);
    oskar_mem_free(model->Q, status);
    oskar_mem_free(model->U, status);
    oskar_mem_free(model->V, status);
    oskar_mem_free(model->reference_freq_hz, status);
    oskar_mem_free(model->spectral_index, status);
    oskar_mem_free(model->rm_rad, status);
    oskar_mem_free(model->l, status);
    oskar_mem_free(model->m, status);
    oskar_mem_free(model->n, status);
    oskar_mem_free(model->fwhm_major_rad, status);
    oskar_mem_free(model->fwhm_minor_rad, status);
    oskar_mem_free(model->pa_rad, status);
    oskar_mem_free(model->gaussian_a, status);
    oskar_mem_free(model->gaussian_b, status);
    oskar_mem_free(model->gaussian_c, status);

    /* Free the structure itself. */
    free(model);
}

#ifdef __cplusplus
}
#endif
