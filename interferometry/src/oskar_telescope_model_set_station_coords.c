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

#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_set_station_coords.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_model_set_station_coords(oskar_TelescopeModel* dst,
        int index, double x, double y, double z,
        double x_hor, double y_hor, double z_hor, int* status)
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

    /* Check range. */
    if (index >= dst->num_stations)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type and location. */
    type = oskar_telescope_model_type(dst);
    location = oskar_telescope_model_location(dst);

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)dst->station_x.data)[index] = x;
            ((double*)dst->station_y.data)[index] = y;
            ((double*)dst->station_z.data)[index] = z;
            ((double*)dst->station_x_hor.data)[index] = x_hor;
            ((double*)dst->station_y_hor.data)[index] = y_hor;
            ((double*)dst->station_z_hor.data)[index] = z_hor;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)dst->station_x.data)[index] = (float)x;
            ((float*)dst->station_y.data)[index] = (float)y;
            ((float*)dst->station_z.data)[index] = (float)z;
            ((float*)dst->station_x_hor.data)[index] = (float)x_hor;
            ((float*)dst->station_y_hor.data)[index] = (float)y_hor;
            ((float*)dst->station_z_hor.data)[index] = (float)z_hor;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        size_t element_size, offset_bytes;
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
            cudaMemcpy((char*)(dst->station_x_hor.data) + offset_bytes, &x_hor,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_y_hor.data) + offset_bytes, &y_hor,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_z_hor.data) + offset_bytes, &z_hor,
                    element_size, cudaMemcpyHostToDevice);
        }
        else if (type == OSKAR_SINGLE)
        {
            float tx, ty, tz, tx_hor, ty_hor, tz_hor;
            tx = (float) x;
            ty = (float) y;
            tz = (float) z;
            tx_hor = (float) x_hor;
            ty_hor = (float) y_hor;
            tz_hor = (float) z_hor;
            cudaMemcpy((char*)(dst->station_x.data) + offset_bytes, &tx,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_y.data) + offset_bytes, &ty,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_z.data) + offset_bytes, &tz,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_x_hor.data) + offset_bytes, &tx_hor,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_y_hor.data) + offset_bytes, &ty_hor,
                    element_size, cudaMemcpyHostToDevice);
            cudaMemcpy((char*)(dst->station_z_hor.data) + offset_bytes, &tz_hor,
                    element_size, cudaMemcpyHostToDevice);
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
