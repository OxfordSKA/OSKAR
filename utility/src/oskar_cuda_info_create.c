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

#include "utility/oskar_CudaInfo.h"
#include "utility/oskar_cuda_info_create.h"
#include "utility/oskar_cuda_device_info_scan.h"
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_cuda_info_create(oskar_CudaInfo** info)
{
    oskar_CudaInfo* inf;
    int error, i;

    /* Allocate index. */
    inf = (oskar_CudaInfo*) malloc(sizeof(oskar_CudaInfo));
    *info = inf;

    /* Get the runtime version and the driver version. */
    cudaDriverGetVersion(&inf->driver_version);
    cudaRuntimeGetVersion(&inf->runtime_version);

    /* Query the number of devices in the system. */
    error = cudaGetDeviceCount(&inf->num_devices);
    if (error != cudaSuccess || inf->num_devices == 0)
    {
        fprintf(stderr, "Unable to determine number of CUDA devices: %s\n",
                cudaGetErrorString(error));
        return error;
    }

    /* Allocate array big enough. */
    inf->device = (oskar_CudaDeviceInfo*) malloc(inf->num_devices *
            sizeof(oskar_CudaDeviceInfo));

    /* Populate device array. */
    for (i = 0; i < inf->num_devices; ++i)
    {
        oskar_cuda_device_info_scan(&(inf->device[i]), i);
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
