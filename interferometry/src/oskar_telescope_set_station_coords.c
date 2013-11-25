/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <private_telescope.h>
#include <oskar_telescope.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords(oskar_Telescope* dst,
        int index, double x, double y, double z,
        double x_hor, double y_hor, double z_hor, int* status)
{
    int type, location;
    char *xw, *yw, *zw, *xh, *yh, *zh;

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
    type = oskar_telescope_precision(dst);
    location = oskar_telescope_location(dst);

    /* Get byte pointers. */
    xw = oskar_mem_char(&dst->station_x);
    yw = oskar_mem_char(&dst->station_y);
    zw = oskar_mem_char(&dst->station_z);
    xh = oskar_mem_char(&dst->station_x_hor);
    yh = oskar_mem_char(&dst->station_y_hor);
    zh = oskar_mem_char(&dst->station_z_hor);

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)xw)[index] = x;
            ((double*)yw)[index] = y;
            ((double*)zw)[index] = z;
            ((double*)xh)[index] = x_hor;
            ((double*)yh)[index] = y_hor;
            ((double*)zh)[index] = z_hor;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)xw)[index] = (float)x;
            ((float*)yw)[index] = (float)y;
            ((float*)zw)[index] = (float)z;
            ((float*)xh)[index] = (float)x_hor;
            ((float*)yh)[index] = (float)y_hor;
            ((float*)zh)[index] = (float)z_hor;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        size_t size, offset_bytes;
        size = oskar_mem_element_size(type);
        offset_bytes = index * size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy(xw + offset_bytes, &x, size, H2D);
            cudaMemcpy(yw + offset_bytes, &y, size, H2D);
            cudaMemcpy(zw + offset_bytes, &z, size, H2D);
            cudaMemcpy(xh + offset_bytes, &x_hor, size, H2D);
            cudaMemcpy(yh + offset_bytes, &y_hor, size, H2D);
            cudaMemcpy(zh + offset_bytes, &z_hor, size, H2D);
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
            cudaMemcpy(xw + offset_bytes, &tx, size, H2D);
            cudaMemcpy(yw + offset_bytes, &ty, size, H2D);
            cudaMemcpy(zw + offset_bytes, &tz, size, H2D);
            cudaMemcpy(xh + offset_bytes, &tx_hor, size, H2D);
            cudaMemcpy(yh + offset_bytes, &ty_hor, size, H2D);
            cudaMemcpy(zh + offset_bytes, &tz_hor, size, H2D);
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
