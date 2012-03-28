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

#include "oskar_global.h"
#include "station/cudak/oskar_cudak_blank_below_horizon.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_blank_below_horizon(oskar_Mem* data, const oskar_Mem* mask)
{
    int type, num_sources;

    /* Sanity check on inputs. */
    if (!mask || !data)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check that all arrays are on the GPU. */
    if (mask->location != OSKAR_LOCATION_GPU ||
            data->location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check that the mask type is OK. */
    if (mask->type != OSKAR_SINGLE && mask->type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that the dimensions are OK. */
    num_sources = mask->num_elements;
    if (data->num_elements < num_sources)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Zero the value of any positions below the horizon. */
    type = data->type;
    if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_blank_below_horizon_matrix_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_sources, (const float*)mask->data, (float4c*)data->data);
    }
    else if (type == OSKAR_SINGLE_COMPLEX)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_blank_below_horizon_scalar_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_sources, (const float*)mask->data, (float2*)data->data);
    }
    else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_blank_below_horizon_matrix_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_sources, (const double*)mask->data, (double4c*)data->data);
    }
    else if (type == OSKAR_DOUBLE_COMPLEX)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_blank_below_horizon_scalar_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_sources, (const double*)mask->data, (double2*)data->data);
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Report any CUDA error. */
    cudaDeviceSynchronize();
    return (int)cudaPeekAtLastError();
}
