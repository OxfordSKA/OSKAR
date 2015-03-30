/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
/* Must include this first to avoid type conflict.*/
#include <cuda_runtime_api.h>
#define H2D cudaMemcpyHostToDevice
#endif

#include <stdlib.h>
#include <oskar_cmath.h>

#include <private_station.h>
#include <oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_errors(oskar_Station* dst,
        int index, double gain, double gain_error, double phase_offset,
        double phase_error, int* status)
{
    int type, location;
    void *gain_, *gain_err_, *phase_, *phase_err_;

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
    type = oskar_mem_type(dst->element_gain);
    if (type != oskar_mem_type(dst->element_gain_error) ||
            type != oskar_mem_type(dst->element_phase_offset_rad) ||
            type != oskar_mem_type(dst->element_phase_error_rad))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get the data location. */
    location = oskar_mem_location(dst->element_gain);
    if (location != oskar_mem_location(dst->element_gain_error) ||
            location != oskar_mem_location(dst->element_phase_offset_rad) ||
            location != oskar_mem_location(dst->element_phase_error_rad))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Get byte pointers. */
    gain_      = oskar_mem_void(dst->element_gain);
    gain_err_  = oskar_mem_void(dst->element_gain_error);
    phase_     = oskar_mem_void(dst->element_phase_offset_rad);
    phase_err_ = oskar_mem_void(dst->element_phase_error_rad);

    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)gain_)[index] = gain;
            ((double*)gain_err_)[index] = gain_error;
            ((double*)phase_)[index] = phase_offset;
            ((double*)phase_err_)[index] = phase_error;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)gain_)[index] = (float) gain;
            ((float*)gain_err_)[index] = (float) gain_error;
            ((float*)phase_)[index] = (float) phase_offset;
            ((float*)phase_err_)[index] = (float) phase_error;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Get the data type. */
        size_t size, offset_bytes;
        size = oskar_mem_element_size(type);
        offset_bytes = index * size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)gain_ + offset_bytes, &gain, size, H2D);
            cudaMemcpy((char*)gain_err_ + offset_bytes, &gain_error, size, H2D);
            cudaMemcpy((char*)phase_ + offset_bytes, &phase_offset, size, H2D);
            cudaMemcpy((char*)phase_err_ + offset_bytes, &phase_error, size, H2D);
        }
        else if (type == OSKAR_SINGLE)
        {
            float t_amp_gain, t_amp_error, t_phase_offset, t_phase_error;
            t_amp_gain = (float) gain;
            t_amp_error = (float) gain_error;
            t_phase_offset = (float) phase_offset;
            t_phase_error = (float) phase_error;
            cudaMemcpy((char*)gain_ + offset_bytes, &t_amp_gain, size, H2D);
            cudaMemcpy((char*)gain_err_ + offset_bytes, &t_amp_error, size, H2D);
            cudaMemcpy((char*)phase_ + offset_bytes, &t_phase_offset, size, H2D);
            cudaMemcpy((char*)phase_err_ + offset_bytes, &t_phase_error, size, H2D);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
