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

/* Must include this first to avoid type conflict.*/
#include <cuda_runtime_api.h>

#include <stdlib.h>

#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_set_element_coords(oskar_StationModel* dst,
        int index, double x, double y, double z, double delta_x,
        double delta_y, double delta_z)
{
    int type, location;
    double x_weights, y_weights, z_weights, x_signal, y_signal, z_signal;

    /* Check range. */
    if (index >= dst->num_elements)
        return OSKAR_ERR_OUT_OF_RANGE;

    /* Get the data type and location. */
    type = oskar_station_model_type(dst);
    location = oskar_station_model_location(dst);

    /* Check if z or delta_z is nonzero, and set 3D flag if so. */
    if (z != 0.0 || delta_z != 0.0)
        dst->array_is_3d = OSKAR_TRUE;

    /* Set up the data. */
    x_weights = x;
    y_weights = y;
    z_weights = z;
    x_signal = x + delta_x;
    y_signal = y + delta_y;
    z_signal = z + delta_z;

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)dst->x_weights.data)[index] = x_weights;
            ((double*)dst->y_weights.data)[index] = y_weights;
            ((double*)dst->z_weights.data)[index] = z_weights;
            ((double*)dst->x_signal.data)[index] = x_signal;
            ((double*)dst->y_signal.data)[index] = y_signal;
            ((double*)dst->z_signal.data)[index] = z_signal;
            return 0;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)dst->x_weights.data)[index] = (float)x_weights;
            ((float*)dst->y_weights.data)[index] = (float)y_weights;
            ((float*)dst->z_weights.data)[index] = (float)z_weights;
            ((float*)dst->x_signal.data)[index] = (float)x_signal;
            ((float*)dst->y_signal.data)[index] = (float)y_signal;
            ((float*)dst->z_signal.data)[index] = (float)z_signal;
            return 0;
        }
        else
            return OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        size_t element_size, offset_bytes;
        element_size = oskar_mem_element_size(type);
        offset_bytes = index * element_size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy((char*)(dst->x_weights.data) + offset_bytes, &x_weights,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y_weights.data) + offset_bytes, &y_weights,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z_weights.data) + offset_bytes, &z_weights,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->x_signal.data) + offset_bytes, &x_signal,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y_signal.data) + offset_bytes, &y_signal,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z_signal.data) + offset_bytes, &z_signal,
                    element_size, cudaMemcpyHostToDevice);
            return 0;
        }
        else if (type == OSKAR_SINGLE)
        {
            float txw, tyw, tzw, txs, tys, tzs;
            txw = (float) x_weights;
            tyw = (float) y_weights;
            tzw = (float) z_weights;
            txs = (float) x_signal;
            tys = (float) y_signal;
            tzs = (float) z_signal;
            cudaMemcpy((char*)(dst->x_weights.data) + offset_bytes, &txw,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y_weights.data) + offset_bytes, &tyw,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z_weights.data) + offset_bytes, &tzw,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->x_signal.data) + offset_bytes, &txs,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->y_signal.data) + offset_bytes, &tys,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->z_signal.data) + offset_bytes, &tzs,
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
