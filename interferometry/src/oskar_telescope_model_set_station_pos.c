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

#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_set_station_pos.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_telescope_model_set_station_pos(oskar_TelescopeModel* dst,
		int index, double x, double y, double z)
{
    if (index >= dst->num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;

    if (oskar_telescope_model_is_location(dst, OSKAR_LOCATION_CPU))
    {
    	if (oskar_telescope_model_is_type(dst, OSKAR_DOUBLE))
    	{
            ((double*)dst->station_x.data)[index] = x;
            ((double*)dst->station_y.data)[index] = y;
            ((double*)dst->station_z.data)[index] = z;
            return 0;
    	}
    	else if (oskar_telescope_model_is_type(dst, OSKAR_SINGLE))
    	{
            ((float*)dst->station_x.data)[index] = (float)x;
            ((float*)dst->station_y.data)[index] = (float)y;
            ((float*)dst->station_z.data)[index] = (float)z;
            return 0;
    	}
    	else
    		return OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (oskar_telescope_model_is_location(dst, OSKAR_LOCATION_GPU))
    {
    	int type;
    	size_t element_size, offset_bytes;

    	/* Get the data type. */
    	type = oskar_telescope_model_type(dst);
        element_size = oskar_mem_element_size(type);
        offset_bytes = index * element_size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)(dst->station_x.data) + offset_bytes, &x,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_y.data) + offset_bytes, &y,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_z.data) + offset_bytes, &z,
            		element_size, cudaMemcpyHostToDevice);
            return 0;
        }
        else if (type == OSKAR_SINGLE)
        {
        	float tx, ty, tz;
        	tx = (float) x;
        	ty = (float) y;
        	tz = (float) z;
            cudaMemcpy((char*)(dst->station_x.data) + offset_bytes, &tx,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_y.data) + offset_bytes, &ty,
            		element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_z.data) + offset_bytes, &tz,
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
