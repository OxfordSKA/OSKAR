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

#include "station/oskar_station_model_set_element_orientation.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_set_element_orientation(oskar_StationModel* dst,
        int index, double orientation_x, double orientation_y, int* status)
{
    int type, location;
    size_t element_size, offset_bytes;

    /* Check all inputs. */
    if (!dst || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check range. */
    if (index >= dst->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type. */
    type = dst->cos_orientation_x.type;
    location = dst->cos_orientation_x.location;
    element_size = oskar_mem_element_size(type);
    offset_bytes = index * element_size;

    /* Convert orientations to radians. */
    orientation_x *= M_PI / 180.0;
    orientation_y *= M_PI / 180.0;
    if (index == 0)
    {
        dst->orientation_x = orientation_x;
        dst->orientation_y = orientation_y;
    }

    /* Check the type. */
    if (type == OSKAR_DOUBLE)
    {
        double cos_x, sin_x, cos_y, sin_y;
        cos_x = cos(orientation_x);
        sin_x = sin(orientation_x);
        cos_y = cos(orientation_y);
        sin_y = sin(orientation_y);

        if (location == OSKAR_LOCATION_CPU)
        {
            ((double*)dst->cos_orientation_x.data)[index] = cos_x;
            ((double*)dst->sin_orientation_x.data)[index] = sin_x;
            ((double*)dst->cos_orientation_y.data)[index] = cos_y;
            ((double*)dst->sin_orientation_y.data)[index] = sin_y;
        }
        else if (location == OSKAR_LOCATION_GPU)
        {
            cudaMemcpy((char*)(dst->cos_orientation_x.data) + offset_bytes,
                    &cos_x, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->sin_orientation_x.data) + offset_bytes,
                    &sin_x, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->cos_orientation_y.data) + offset_bytes,
                    &cos_y, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->sin_orientation_y.data) + offset_bytes,
                    &sin_y, element_size, cudaMemcpyHostToDevice);
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else if (type == OSKAR_SINGLE)
    {
        float cos_x, sin_x, cos_y, sin_y;
        cos_x = (float) cos(orientation_x);
        sin_x = (float) sin(orientation_x);
        cos_y = (float) cos(orientation_y);
        sin_y = (float) sin(orientation_y);

        if (location == OSKAR_LOCATION_CPU)
        {
            ((float*)dst->cos_orientation_x.data)[index] = cos_x;
            ((float*)dst->sin_orientation_x.data)[index] = sin_x;
            ((float*)dst->cos_orientation_y.data)[index] = cos_y;
            ((float*)dst->sin_orientation_y.data)[index] = sin_y;
        }
        else if (location == OSKAR_LOCATION_GPU)
        {
            cudaMemcpy((char*)(dst->cos_orientation_x.data) + offset_bytes,
                    &cos_x, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->sin_orientation_x.data) + offset_bytes,
                    &sin_x, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->cos_orientation_y.data) + offset_bytes,
                    &cos_y, element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->sin_orientation_y.data) + offset_bytes,
                    &sin_y, element_size, cudaMemcpyHostToDevice);
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
