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

#include "station/oskar_station_model_set_element_pos.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_set_element_pos(oskar_StationModel* dst,
		int index, double x, double y, double z)
{
	int type, location;

	/* Check range. */
    if (index >= dst->n_elements)
        return OSKAR_ERR_OUT_OF_RANGE;

	/* Get the data type. */
    if (dst->x.private_type != dst->y.private_type ||
    		dst->x.private_type != dst->z.private_type)
    	return OSKAR_ERR_TYPE_MISMATCH;
	type = dst->x.private_type;

	/* Get the data location. */
    if (dst->x.private_location != dst->y.private_location ||
    		dst->x.private_location != dst->z.private_location)
    	return OSKAR_ERR_BAD_LOCATION;
	location = dst->x.private_location;

    if (location == OSKAR_LOCATION_CPU)
    {
    	if (type == OSKAR_DOUBLE)
    	{
            ((double*)dst->x.data)[index] = x;
            ((double*)dst->y.data)[index] = y;
            ((double*)dst->z.data)[index] = z;
            return 0;
    	}
    	else if (type == OSKAR_SINGLE)
    	{
            ((float*)dst->x.data)[index] = (float)x;
            ((float*)dst->y.data)[index] = (float)y;
            ((float*)dst->z.data)[index] = (float)z;
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
            cudaMemcpy((char*)(dst->x.data) + offset_bytes, &x,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y.data) + offset_bytes, &y,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z.data) + offset_bytes, &z,
            		element_size, cudaMemcpyHostToDevice);
            return 0;
        }
        else if (type == OSKAR_SINGLE)
        {
        	float tx, ty, tz;
        	tx = (float) x;
        	ty = (float) y;
        	tz = (float) z;
            cudaMemcpy((char*)(dst->x.data) + offset_bytes, &tx,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y.data) + offset_bytes, &ty,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z.data) + offset_bytes, &tz,
            		element_size, cudaMemcpyHostToDevice);
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
