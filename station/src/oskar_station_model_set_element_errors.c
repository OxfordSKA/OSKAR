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

/* Must include this first to avoid type conflict. */
#include <cuda_runtime_api.h>

#include <stdlib.h>
#include <math.h>

#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_set_element_errors(oskar_StationModel* dst,
        int index, double gain, double gain_error, double phase_offset,
        double phase_error, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!dst || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert phases to radians */
    phase_offset *= M_PI / 180.0;
    phase_error *= M_PI / 180.0;

    /* Check range. */
    if (index >= dst->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type. */
    type = dst->gain.type;
    if (type != dst->gain_error.type || type != dst->phase_offset.type ||
            type != dst->phase_error.type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get the data location. */
    location = dst->gain.location;
    if (location != dst->gain_error.location ||
            location != dst->phase_offset.location ||
            location != dst->phase_error.location)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)dst->gain.data)[index] = gain;
            ((double*)dst->gain_error.data)[index] = gain_error;
            ((double*)dst->phase_offset.data)[index] = phase_offset;
            ((double*)dst->phase_error.data)[index] = phase_error;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)dst->gain.data)[index] = (float) gain;
            ((float*)dst->gain_error.data)[index] = (float) gain_error;
            ((float*)dst->phase_offset.data)[index] = (float) phase_offset;
            ((float*)dst->phase_error.data)[index] = (float) phase_error;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        /* Get the data type. */
        size_t element_size, offset_bytes;
        element_size = oskar_mem_element_size(type);
        offset_bytes = index * element_size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)(dst->gain.data) + offset_bytes,
                    &gain, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->gain_error.data) + offset_bytes,
                    &gain_error, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_offset.data) + offset_bytes,
                    &phase_offset, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_error.data) + offset_bytes,
                    &phase_error, element_size, cudaMemcpyHostToDevice);
        }
        else if (type == OSKAR_SINGLE)
        {
            float t_amp_gain, t_amp_error, t_phase_offset, t_phase_error;
            t_amp_gain = (float) gain;
            t_amp_error = (float) gain_error;
            t_phase_offset = (float) phase_offset;
            t_phase_error = (float) phase_error;
            cudaMemcpy((char*)(dst->gain.data) + offset_bytes,
                    &t_amp_gain, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->gain_error.data) + offset_bytes,
                    &t_amp_error, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_offset.data) + offset_bytes,
                    &t_phase_offset, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_error.data) + offset_bytes,
                    &t_phase_error, element_size, cudaMemcpyHostToDevice);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
