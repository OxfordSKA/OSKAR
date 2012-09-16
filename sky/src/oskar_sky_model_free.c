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

#include "sky/oskar_sky_model_free.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_model_free(oskar_SkyModel* model, int* status)
{
    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Initialise the memory. */
    oskar_mem_free(&model->RA, status);
    oskar_mem_free(&model->Dec, status);
    oskar_mem_free(&model->I, status);
    oskar_mem_free(&model->Q, status);
    oskar_mem_free(&model->U, status);
    oskar_mem_free(&model->V, status);
    oskar_mem_free(&model->reference_freq, status);
    oskar_mem_free(&model->spectral_index, status);
    oskar_mem_free(&model->rel_l, status);
    oskar_mem_free(&model->rel_m, status);
    oskar_mem_free(&model->rel_n, status);
    oskar_mem_free(&model->FWHM_major, status);
    oskar_mem_free(&model->FWHM_minor, status);
    oskar_mem_free(&model->position_angle, status);
    oskar_mem_free(&model->gaussian_a, status);
    oskar_mem_free(&model->gaussian_b, status);
    oskar_mem_free(&model->gaussian_c, status);

    /* Set meta-data */
    model->num_sources = 0;
    model->use_extended = OSKAR_FALSE;
}

#ifdef __cplusplus
}
#endif
