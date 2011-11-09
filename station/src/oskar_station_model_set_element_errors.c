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

#include <cuda_runtime_api.h> // Must include this first to avoid type conflict.
#include <stdlib.h>

#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_set_element_errors(oskar_StationModel* dst,
		int index, double amp_gain, double amp_error, double phase_offset,
		double phase_error)
{
	int type, location;

	/* Check range. */
    if (index >= dst->num_elements)
        return OSKAR_ERR_OUT_OF_RANGE;

	/* Get the data type. */
    if (dst->amp_gain.private_type != dst->amp_error.private_type ||
    		dst->amp_gain.private_type != dst->phase_offset.private_type ||
    		dst->amp_gain.private_type != dst->phase_error.private_type)
    	return OSKAR_ERR_TYPE_MISMATCH;
	type = dst->amp_gain.private_type;

	/* Get the data location. */
    if (dst->amp_gain.private_location != dst->amp_error.private_location ||
    		dst->amp_gain.private_location != dst->phase_offset.private_location ||
    		dst->amp_gain.private_location != dst->phase_error.private_location)
    	return OSKAR_ERR_BAD_LOCATION;
	location = dst->amp_gain.private_location;

    if (location == OSKAR_LOCATION_CPU)
    {
    	if (type == OSKAR_DOUBLE)
    	{
            ((double*)dst->amp_gain.data)[index] = amp_gain;
            ((double*)dst->amp_error.data)[index] = amp_error;
            ((double*)dst->phase_offset.data)[index] = phase_offset;
            ((double*)dst->phase_error.data)[index] = phase_error;
            return 0;
    	}
    	else if (type == OSKAR_SINGLE)
    	{
            ((float*)dst->amp_gain.data)[index] = (float) amp_gain;
            ((float*)dst->amp_error.data)[index] = (float) amp_error;
            ((float*)dst->phase_offset.data)[index] = (float) phase_offset;
            ((float*)dst->phase_error.data)[index] = (float) phase_error;
            return 0;
    	}
    	else
    		return OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
    	/* Get the data type. */
    	size_t element_size, offset_bytes;
        element_size = oskar_mem_element_size(type);
        offset_bytes = index * element_size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)(dst->amp_gain.data) + offset_bytes,
            		&amp_gain, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->amp_error.data) + offset_bytes,
            		&amp_error, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_offset.data) + offset_bytes,
            		&phase_offset, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_error.data) + offset_bytes,
            		&phase_error, element_size, cudaMemcpyHostToDevice);
            return 0;
        }
        else if (type == OSKAR_SINGLE)
        {
        	float t_amp_gain, t_amp_error, t_phase_offset, t_phase_error;
        	t_amp_gain = (float) amp_gain;
        	t_amp_error = (float) amp_error;
        	t_phase_offset = (float) phase_offset;
        	t_phase_error = (float) phase_error;
            cudaMemcpy((char*)(dst->amp_gain.data) + offset_bytes,
            		&t_amp_gain, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->amp_error.data) + offset_bytes,
            		&t_amp_error, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_offset.data) + offset_bytes,
            		&t_phase_offset, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->phase_error.data) + offset_bytes,
            		&t_phase_error, element_size, cudaMemcpyHostToDevice);
            return 0;
        }
        else
    		return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
