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

#include "station/oskar_station_model_set_element_weight.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_set_element_weight(oskar_StationModel* dst,
		int index, double re, double im)
{
	int type, location;
	size_t element_size, offset_bytes;

	/* Check range. */
    if (index >= dst->num_elements)
        return OSKAR_ERR_OUT_OF_RANGE;

	/* Get the data type. */
	type = dst->weight.private_type;
	location = dst->weight.private_location;
    element_size = oskar_mem_element_size(type);
    offset_bytes = index * element_size;

    /* Check the type. */
    if (type == OSKAR_DOUBLE_COMPLEX)
    {
    	double2 w;
    	w.x = re; w.y = im;

    	if (location == OSKAR_LOCATION_CPU)
    		((double2*)dst->weight.data)[index] = w;
    	else if (location == OSKAR_LOCATION_GPU)
            cudaMemcpy((char*)(dst->weight.data) + offset_bytes, &w,
            		element_size, cudaMemcpyHostToDevice);
    	else
    		return OSKAR_ERR_BAD_LOCATION;
    }
    else if (type == OSKAR_SINGLE_COMPLEX)
    {
    	float2 w;
    	w.x = (float)re; w.y = (float)im;

    	if (location == OSKAR_LOCATION_CPU)
    		((float2*)dst->weight.data)[index] = w;
    	else if (location == OSKAR_LOCATION_GPU)
            cudaMemcpy((char*)(dst->weight.data) + offset_bytes, &w,
            		element_size, cudaMemcpyHostToDevice);
    	else
    		return OSKAR_ERR_BAD_LOCATION;
    }
    else
    	return OSKAR_ERR_BAD_DATA_TYPE;

    return 0;
}

#ifdef __cplusplus
}
#endif
