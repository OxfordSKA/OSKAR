/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    sky list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    sky list of conditions and the following disclaimer in the documentation
 *    and/or src materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from sky
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

#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_type.h"
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_Mem.h"
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_set_source(oskar_SkyModel* sky, int index, double ra,
        double dec, double I, double Q, double U, double V, double ref_frequency,
        double spectral_index)
{
    int type, location;
    if (index >= sky->num_sources)
        return OSKAR_ERR_OUT_OF_RANGE;

    /* Get the data location and type. */
    location = oskar_sky_model_location(sky);
    type = oskar_sky_model_type(sky);

    if (location == OSKAR_LOCATION_GPU)
    {
        size_t element_size, offset_bytes;
        element_size = oskar_mem_element_size(type);
        offset_bytes = index * element_size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)(sky->RA.data) + offset_bytes, &ra, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->Dec.data) + offset_bytes, &dec, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->I.data) + offset_bytes, &I, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->Q.data) + offset_bytes, &Q, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->U.data) + offset_bytes, &U, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->V.data) + offset_bytes, &V, element_size,
                    cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->reference_freq.data) + offset_bytes,
                    &ref_frequency, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->spectral_index.data) + offset_bytes,
                    &spectral_index, element_size, cudaMemcpyHostToDevice);
        }
        else if (type == OSKAR_SINGLE)
        {
            float temp_ra = (float)ra;
            float temp_dec = (float)dec;
            float temp_I = (float)I;
            float temp_Q = (float)Q;
            float temp_U = (float)U;
            float temp_V = (float)V;
            float temp_ref_freq = (float)ref_frequency;
            float temp_spectral_index = (float)spectral_index;
            cudaMemcpy((char*)(sky->RA.data) + offset_bytes, &temp_ra,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->Dec.data) + offset_bytes, &temp_dec,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->I.data) + offset_bytes, &temp_I,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->Q.data) + offset_bytes, &temp_Q,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->U.data) + offset_bytes, &temp_U,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->V.data) + offset_bytes, &temp_V,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->reference_freq.data) + offset_bytes,
                    &temp_ref_freq, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(sky->spectral_index.data) + offset_bytes,
                    &temp_spectral_index, element_size, cudaMemcpyHostToDevice);
        }
    }
    else
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)sky->RA.data)[index] = ra;
            ((double*)sky->Dec.data)[index] = dec;
            ((double*)sky->I.data)[index] = I;
            ((double*)sky->Q.data)[index] = Q;
            ((double*)sky->U.data)[index] = U;
            ((double*)sky->V.data)[index] = V;
            ((double*)sky->reference_freq.data)[index] = ref_frequency;
            ((double*)sky->spectral_index.data)[index] = spectral_index;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)sky->RA.data)[index] = (float)ra;
            ((float*)sky->Dec.data)[index] = (float)dec;
            ((float*)sky->I.data)[index] = (float)I;
            ((float*)sky->Q.data)[index] = (float)Q;
            ((float*)sky->U.data)[index] = (float)U;
            ((float*)sky->V.data)[index] = (float)V;
            ((float*)sky->reference_freq.data)[index] = (float)ref_frequency;
            ((float*)sky->spectral_index.data)[index] = (float)spectral_index;
        }
        else
        {
            return OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
