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

#include "utility/oskar_binary_stream_write_cuda_info.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_CudaInfo.h"
#include <oskar_mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_binary_stream_write_cuda_info(FILE* stream,
        const oskar_CudaInfo* cuda_info, int* status)
{
    int i;

    /* Write the number of devices. */
    oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_NUM_DEVICES, 0,
            cuda_info->num_devices, status);

    /* Write the driver version. */
    oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DRIVER_VERSION, 0,
            cuda_info->driver_version, status);

    /* Write the runtime version. */
    oskar_binary_stream_write_int(stream,
            OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_RUNTIME_VERSION, 0,
            cuda_info->runtime_version, status);

    /* Loop over each device in the structure. */
    for (i = 0; i < cuda_info->num_devices; ++i)
    {
        size_t len;

        /* Write device name. */
        len = 1 + strlen(cuda_info->device[i].name);
        oskar_binary_stream_write(stream, OSKAR_CHAR,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_NAME,
                i, len, cuda_info->device[i].name, status);

        /* Write the compute capability. */
        oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_COMPUTE,
                i, 2 * sizeof(int), cuda_info->device[i].compute.version, status);

        /* Write the global memory size. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_SIZE,
                i, cuda_info->device[i].global_memory_size, status);

        /* Write the number of multiprocessors. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MULTIPROCESSORS,
                i, cuda_info->device[i].num_multiprocessors, status);

        /* Write the GPU clock frequency. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_GPU_CLOCK,
                i, cuda_info->device[i].gpu_clock, status);

        /* Write the memory clock frequency. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_CLOCK,
                i, cuda_info->device[i].memory_clock, status);

        /* Write the memory bus width. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MEMORY_BUS,
                i, cuda_info->device[i].memory_bus_width, status);

        /* Write the level 2 cache size. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_L2_CACHE,
                i, cuda_info->device[i].level_2_cache_size, status);

        /* Write the shared memory size. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_SHARED_MEMORY_SIZE,
                i, cuda_info->device[i].shared_memory_size, status);

        /* Write the registers per block. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_REGS_PER_BLOCK,
                i, cuda_info->device[i].num_registers, status);

        /* Write the warp size. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_WARP_SIZE,
                i, cuda_info->device[i].warp_size, status);

        /* Write the maximum number of threads per block. */
        oskar_binary_stream_write_int(stream,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_PER_BLOCK,
                i, cuda_info->device[i].max_threads_per_block, status);

        /* Write the maximum thread block dimensions. */
        oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_THREADS_DIM,
                i, 3 * sizeof(int), cuda_info->device[i].max_threads_dim, status);

        /* Write the maximum grid dimensions. */
        oskar_binary_stream_write(stream, OSKAR_INT,
                OSKAR_TAG_GROUP_CUDA_INFO, OSKAR_TAG_CUDA_INFO_DEVICE_MAX_GRID_SIZE,
                i, 3 * sizeof(int), cuda_info->device[i].max_grid_size, status);
    }
}

#ifdef __cplusplus
}
#endif
