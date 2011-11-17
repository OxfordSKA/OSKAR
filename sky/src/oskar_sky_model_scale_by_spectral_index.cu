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

#include "sky/oskar_sky_model_scale_by_spectral_index.h"
#include "sky/cudak/oskar_cudak_scale_brightness_by_spectral_index.h"
#include <cstdio>

extern "C"
int oskar_sky_model_scale_by_spectral_index(oskar_SkyModel* model,
        double frequency)
{
    // Check for sane inputs.
    if (model == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check for the correct location.
    if (model->location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Get the type and dimensions.
    int type = model->type();
    int num_sources = model->num_sources;

    // Scale the brightnesses.
    if (type == OSKAR_SINGLE)
    {
        int num_threads = 256;
        int num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_scale_brightness_by_spectral_index_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                        frequency, model->reference_freq,
                        model->spectral_index, model->I, model->Q,
                        model->U, model->V);
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
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
