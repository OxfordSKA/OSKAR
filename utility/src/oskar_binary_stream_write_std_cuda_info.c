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

#include "utility/oskar_binary_stream_write_std_cuda_info.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_CudaInfo.h"
#include "utility/oskar_Mem.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_binary_stream_write_std_cuda_info(FILE* stream,
        const oskar_CudaInfo* cuda_info)
{
    int i, error;

    /* Write the number of devices. */
    error = oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_NUM_DEVICES, 0,
            cuda_info->num_devices);
    if (error) return error;

    /* Write the driver version. */
    error = oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DRIVER_VERSION, 0,
            cuda_info->driver_version);
    if (error) return error;

    /* Write the runtime version. */
    error = oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_RUNTIME_VERSION, 0,
            cuda_info->runtime_version);
    if (error) return error;

    /* Loop over each device in the structure. */
    for (i = 0; i < cuda_info->num_devices; ++i)
    {
        size_t len;

        /* Write device name. */
        len = 1 + strlen(cuda_info->device[i].name);
        error = oskar_binary_stream_write(stream, OSKAR_CHAR,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_NAME,
                i, len, cuda_info->device[i].name);
        if (error) return error;

        /* Write the compute capability. */
        error = oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_COMPUTE,
                i, 2 * sizeof(int), cuda_info->device[i].compute.version);
        if (error) return error;

        /* Write the global memory size. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_SIZE,
                i, cuda_info->device[i].global_memory_size);
        if (error) return error;

        /* Write the number of multiprocessors. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MULTIPROCESSORS,
                i, cuda_info->device[i].num_multiprocessors);
        if (error) return error;

        /* Write the GPU clock frequency. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_GPU_CLOCK,
                i, cuda_info->device[i].gpu_clock);
        if (error) return error;

        /* Write the memory clock frequency. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_CLOCK,
                i, cuda_info->device[i].memory_clock);
        if (error) return error;

        /* Write the memory bus width. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_BUS,
                i, cuda_info->device[i].memory_bus_width);
        if (error) return error;

        /* Write the level 2 cache size. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_L2_CACHE,
                i, cuda_info->device[i].level_2_cache_size);
        if (error) return error;

        /* Write the shared memory size. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_SHARED_MEMORY_SIZE,
                i, cuda_info->device[i].shared_memory_size);
        if (error) return error;

        /* Write the registers per block. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_REGS_PER_BLOCK,
                i, cuda_info->device[i].num_registers);
        if (error) return error;

        /* Write the warp size. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_WARP_SIZE,
                i, cuda_info->device[i].warp_size);
        if (error) return error;

        /* Write the maximum number of threads per block. */
        error = oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_PER_BLOCK,
                i, cuda_info->device[i].max_threads_per_block);
        if (error) return error;

        /* Write the maximum thread block dimensions. */
        error = oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_DIM,
                i, 3 * sizeof(int), cuda_info->device[i].max_threads_dim);

        /* Write the maximum grid dimensions. */
        error = oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_GRID_SIZE,
                i, 3 * sizeof(int), cuda_info->device[i].max_grid_size);
        if (error) return error;
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
