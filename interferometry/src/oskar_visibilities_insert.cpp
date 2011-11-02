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

#include "oskar_global.h"
#include "interferometry/oskar_visibilities_insert.h"
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_Mem.h"
#include <cstring>
#include <cuda_runtime_api.h>

int oskar_visibilties_insert(oskar_Visibilities* dst,
        const oskar_Visibilities* src, int time_index)
{
    int error = 0;

    if (dst == NULL || src == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Restrict the time index that can be inserted into to make sure
    // that the can be completely filled with zero overlap.
    // - is this check needed?
    if (time_index % src->num_times != 0)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (dst->amplitude.type() != src->amplitude.type())
        return OSKAR_ERR_BAD_DATA_TYPE;

    if (dst->baseline_u.type() != src->baseline_u.type())
        return OSKAR_ERR_BAD_DATA_TYPE;

    if (dst->num_baselines != src-> num_baselines)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (dst->num_channels != src-> num_channels)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if ((time_index + src->num_times) > dst->num_times)
        return OSKAR_ERR_OUT_OF_RANGE;

    int num_baselines    = dst->num_baselines;
    int num_channels     = dst->num_channels;
    int samples_per_time = num_baselines * num_channels;
    int index            = time_index * samples_per_time;
    size_t element_size_uvw = oskar_mem_element_size(dst->baseline_u.type());
    size_t element_size_amp = oskar_mem_element_size(dst->amplitude.type());

    cudaMemcpyKind mem_copy_kind;
    if (dst->location() == OSKAR_LOCATION_CPU)
    {
        if (src->location() == OSKAR_LOCATION_CPU)
            mem_copy_kind = cudaMemcpyHostToHost;
        else
            mem_copy_kind = cudaMemcpyDeviceToHost;
    }
    else
    {
        if (src->location() == OSKAR_LOCATION_CPU)
            mem_copy_kind = cudaMemcpyHostToDevice;
        else
            mem_copy_kind = cudaMemcpyDeviceToDevice;
    }

    size_t offset = index * element_size_uvw;
    size_t mem_size = src->num_samples() * element_size_uvw;
    cudaMemcpy((char*)(dst->baseline_u.data) + offset,
            src->baseline_u.data, mem_size, mem_copy_kind);
    cudaMemcpy((char*)(dst->baseline_v.data) + offset,
            src->baseline_v.data, mem_size, mem_copy_kind);
    cudaMemcpy((char*)(dst->baseline_w.data) + offset,
            src->baseline_w.data, mem_size, mem_copy_kind);
    offset   = index * element_size_amp;
    mem_size = src->num_samples() * element_size_amp;
    cudaMemcpy((char*)(dst->amplitude.data) + offset,
            src->amplitude.data, mem_size, mem_copy_kind);
    error = (int)cudaPeekAtLastError();

    return error;
}



