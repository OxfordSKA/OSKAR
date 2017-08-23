/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "utility/private_cuda_info.h"
#include "utility/oskar_cuda_info_create.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_cuda_device_info_scan(oskar_CudaDeviceInfo* device, int id);

oskar_CudaInfo* oskar_cuda_info_create(int* status)
{
    oskar_CudaInfo* info;
    int i;

    /* Allocate index. */
    info = (oskar_CudaInfo*) calloc(1, sizeof(oskar_CudaInfo));

    /* Get the runtime version and the driver version. */
    cudaDriverGetVersion(&info->driver_version);
    cudaRuntimeGetVersion(&info->runtime_version);

    /* Query the number of devices in the system. */
    *status = cudaGetDeviceCount(&info->num_devices);
    if (*status != cudaSuccess || info->num_devices == 0)
    {
        fprintf(stderr, "Unable to determine number of CUDA devices: %s\n",
                cudaGetErrorString((cudaError_t)(*status)));
        return info;
    }

    /* Allocate array big enough. */
    info->device = (oskar_CudaDeviceInfo*) calloc(info->num_devices,
            sizeof(oskar_CudaDeviceInfo));

    /* Populate device array. */
    for (i = 0; i < info->num_devices; ++i)
    {
        oskar_cuda_device_info_scan(&(info->device[i]), i);
    }
    return info;
}

void oskar_cuda_device_info_scan(oskar_CudaDeviceInfo* device, int id)
{
    int arch, device_count = 0;
    cudaError_t error;
    struct cudaDeviceProp device_prop;
    size_t total_memory = 0, free_memory = 0;

    /* Set CUDA device. */
    cudaSetDevice(id);

    /* Set default values in case of errors. */
    device->name[0] = 0;
    device->compute.capability.major = 0;
    device->compute.capability.minor = 0;
    device->supports_double = 0;
    device->global_memory_size = 0;
    device->free_memory = 0;
    device->num_multiprocessors = 0;
    device->num_cores = 0;
    device->gpu_clock = 0;
    device->memory_clock = 0;
    device->memory_bus_width = 0;
    device->level_2_cache_size = 0;
    device->shared_memory_size = 0;
    device->num_registers = 0;
    device->warp_size = 0;
    device->max_threads_per_block = 0;
    device->max_threads_dim[0] = 0;
    device->max_threads_dim[1] = 0;
    device->max_threads_dim[2] = 0;
    device->max_grid_size[0] = 0;
    device->max_grid_size[1] = 0;
    device->max_grid_size[2] = 0;

    /* Get device count. */
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0)
    {
        fprintf(stderr, "Unable to determine number of CUDA devices: %s\n",
                cudaGetErrorString(error));
        return;
    }

    /* Check device ID is within range. */
    if (id > device_count - 1)
    {
        fprintf(stderr, "Error: Device ID out of range.\n");
        return;
    }

    /* Get device properties. */
    cudaGetDeviceProperties(&device_prop, id);
    strcpy(device->name, device_prop.name);
    device->compute.capability.major = device_prop.major;
    device->compute.capability.minor = device_prop.minor;
    device->supports_double = 0;
    if (device_prop.major >= 2 || device_prop.minor >= 3)
        device->supports_double = 1;
    total_memory = device_prop.totalGlobalMem / 1024;
    device->global_memory_size = total_memory;
    device->num_multiprocessors = device_prop.multiProcessorCount;
    arch = (device_prop.major << 4) + device_prop.minor;
    switch (arch)
    {
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
        device->num_cores = 8;
        break;
    case 0x20:
        device->num_cores = 32;
        break;
    case 0x21:
        device->num_cores = 48;
        break;
    case 0x30:
    case 0x32:
    case 0x35:
    case 0x37:
        device->num_cores = 192;
        break;
    case 0x50:
    case 0x52:
    case 0x53:
        device->num_cores = 128;
        break;
    case 0x60:
        device->num_cores = 64;
        break;
    case 0x61:
    case 0x62:
        device->num_cores = 128;
        break;
    case 0x70:
        device->num_cores = 64;
        break;
    default:
        device->num_cores = -1;
        break;
    }
    if (device->num_cores > 0)
        device->num_cores *= device->num_multiprocessors;
    device->gpu_clock = device_prop.clockRate;
#if CUDART_VERSION >= 4000
    device->memory_clock = device_prop.memoryClockRate;
    device->memory_bus_width = device_prop.memoryBusWidth;
    device->level_2_cache_size = device_prop.l2CacheSize;
#else
    device->memory_clock = -1;
    device->memory_bus_width = -1;
    device->level_2_cache_size = -1;
#endif

    /* Get free memory size. */
    cudaMemGetInfo(&free_memory, &total_memory);
    free_memory /= 1024;
    device->free_memory = free_memory;

    /* Get block properties. */
    device->shared_memory_size = device_prop.sharedMemPerBlock;
    device->num_registers = device_prop.regsPerBlock;
    device->warp_size = device_prop.warpSize;
    device->max_threads_per_block = device_prop.maxThreadsPerBlock;
    device->max_threads_dim[0] = device_prop.maxThreadsDim[0];
    device->max_threads_dim[1] = device_prop.maxThreadsDim[1];
    device->max_threads_dim[2] = device_prop.maxThreadsDim[2];
    device->max_grid_size[0] = device_prop.maxGridSize[0];
    device->max_grid_size[1] = device_prop.maxGridSize[1];
    device->max_grid_size[2] = device_prop.maxGridSize[2];
}

#ifdef __cplusplus
}
#endif
