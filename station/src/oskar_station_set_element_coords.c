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

#include <private_station.h>
#include <oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_coords(oskar_Station* dst,
        int index, const double measured_enu[3], const double true_enu[3],
        int* status)
{
    int type, location;
    void *xw, *yw, *zw, *xs, *ys, *zs;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check range. */
    if (index >= dst->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type and location. */
    type = oskar_station_precision(dst);
    location = oskar_station_mem_location(dst);

    /* Check if any z component is nonzero, and set 3D flag if so. */
    if (measured_enu[2] != 0.0 || true_enu[2] != 0.0)
        dst->array_is_3d = OSKAR_TRUE;

    /* Get raw pointers. */
    xw = oskar_mem_void(dst->element_measured_x_enu_metres);
    yw = oskar_mem_void(dst->element_measured_y_enu_metres);
    zw = oskar_mem_void(dst->element_measured_z_enu_metres);
    xs = oskar_mem_void(dst->element_true_x_enu_metres);
    ys = oskar_mem_void(dst->element_true_y_enu_metres);
    zs = oskar_mem_void(dst->element_true_z_enu_metres);

    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            ((double*)xw)[index] = measured_enu[0];
            ((double*)yw)[index] = measured_enu[1];
            ((double*)zw)[index] = measured_enu[2];
            ((double*)xs)[index] = true_enu[0];
            ((double*)ys)[index] = true_enu[1];
            ((double*)zs)[index] = true_enu[2];
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)xw)[index] = (float)measured_enu[0];
            ((float*)yw)[index] = (float)measured_enu[1];
            ((float*)zw)[index] = (float)measured_enu[2];
            ((float*)xs)[index] = (float)true_enu[0];
            ((float*)ys)[index] = (float)true_enu[1];
            ((float*)zs)[index] = (float)true_enu[2];
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
            cudaMemcpy((char*)xw + offset_bytes, &measured_enu[0], size, H2D);
            cudaMemcpy((char*)yw + offset_bytes, &measured_enu[1], size, H2D);
            cudaMemcpy((char*)zw + offset_bytes, &measured_enu[2], size, H2D);
            cudaMemcpy((char*)xs + offset_bytes, &true_enu[0], size, H2D);
            cudaMemcpy((char*)ys + offset_bytes, &true_enu[1], size, H2D);
            cudaMemcpy((char*)zs + offset_bytes, &true_enu[2], size, H2D);
        }
        else if (type == OSKAR_SINGLE)
        {
            float txw, tyw, tzw, txs, tys, tzs;
            txw = (float) measured_enu[0];
            tyw = (float) measured_enu[1];
            tzw = (float) measured_enu[2];
            txs = (float) true_enu[0];
            tys = (float) true_enu[1];
            tzs = (float) true_enu[2];
            cudaMemcpy((char*)xw + offset_bytes, &txw, size, H2D);
            cudaMemcpy((char*)yw + offset_bytes, &tyw, size, H2D);
            cudaMemcpy((char*)zw + offset_bytes, &tzw, size, H2D);
            cudaMemcpy((char*)xs + offset_bytes, &txs, size, H2D);
            cudaMemcpy((char*)ys + offset_bytes, &tys, size, H2D);
            cudaMemcpy((char*)zs + offset_bytes, &tzs, size, H2D);
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
