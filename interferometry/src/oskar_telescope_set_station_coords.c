/*
 * Copyright (c) 2011-2014, The University of Oxford
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

void oskar_telescope_set_station_coords(oskar_Telescope* dst, int index,
        double x_offset_ecef, double y_offset_ecef, double z_offset_ecef,
        double x_enu, double y_enu, double z_enu, int* status)
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
    location = oskar_telescope_mem_location(dst);

    /* Get byte pointers. */
    xw = oskar_mem_char(dst->station_true_x_offset_ecef_metres);
    yw = oskar_mem_char(dst->station_true_y_offset_ecef_metres);
    zw = oskar_mem_char(dst->station_true_z_offset_ecef_metres);
    xh = oskar_mem_char(dst->station_true_x_enu_metres);
    yh = oskar_mem_char(dst->station_true_y_enu_metres);
    zh = oskar_mem_char(dst->station_true_z_enu_metres);

    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)xw)[index] = x_offset_ecef;
            ((double*)yw)[index] = y_offset_ecef;
            ((double*)zw)[index] = z_offset_ecef;
            ((double*)xh)[index] = x_enu;
            ((double*)yh)[index] = y_enu;
            ((double*)zh)[index] = z_enu;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)xw)[index] = (float)x_offset_ecef;
            ((float*)yw)[index] = (float)y_offset_ecef;
            ((float*)zw)[index] = (float)z_offset_ecef;
            ((float*)xh)[index] = (float)x_enu;
            ((float*)yh)[index] = (float)y_enu;
            ((float*)zh)[index] = (float)z_enu;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        size_t size, offset_bytes;
        size = oskar_mem_element_size(type);
        offset_bytes = index * size;
        if (type == OSKAR_DOUBLE)
        {
            cudaMemcpy(xw + offset_bytes, &x_offset_ecef, size, H2D);
            cudaMemcpy(yw + offset_bytes, &y_offset_ecef, size, H2D);
            cudaMemcpy(zw + offset_bytes, &z_offset_ecef, size, H2D);
            cudaMemcpy(xh + offset_bytes, &x_enu, size, H2D);
            cudaMemcpy(yh + offset_bytes, &y_enu, size, H2D);
            cudaMemcpy(zh + offset_bytes, &z_enu, size, H2D);
        }
        else if (type == OSKAR_SINGLE)
        {
            float tx, ty, tz, tx_hor, ty_hor, tz_hor;
            tx = (float) x_offset_ecef;
            ty = (float) y_offset_ecef;
            tz = (float) z_offset_ecef;
            tx_hor = (float) x_enu;
            ty_hor = (float) y_enu;
            tz_hor = (float) z_enu;
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
