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

#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_scale_by_spectral_index.h"
#include "sky/oskar_sky_model_type.h"
#include "sky/cudak/oskar_cudak_scale_brightness_by_spectral_index.h"
#include "utility/oskar_cuda_check_error.h"
#include <cstdio>

extern "C"
void oskar_sky_model_scale_by_spectral_index(oskar_SkyModel* model,
        double frequency, int* status)
{
    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check for the correct location. */
    if (oskar_sky_model_location(model) != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get the type and dimensions. */
    int type = oskar_sky_model_type(model);
    int num_sources = model->num_sources;

    /* Scale the brightnesses. */
    if (type == OSKAR_SINGLE)
    {
        int num_threads = 256;
        int num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_scale_brightness_by_spectral_index_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                        frequency, model->reference_freq,
                        model->spectral_index, model->I, model->Q,
                        model->U, model->V);
        oskar_cuda_check_error(status);
    }
    else if (type == OSKAR_DOUBLE)
    {
        int num_threads = 256;
        int num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_scale_brightness_by_spectral_index_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                        frequency, model->reference_freq,
                        model->spectral_index, model->I, model->Q,
                        model->U, model->V);
        oskar_cuda_check_error(status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}
