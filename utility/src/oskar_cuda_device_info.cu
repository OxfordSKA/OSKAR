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

#include "utility/oskar_cuda_device_info.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_get_cuda_arch(const int deviceId, int* major, int* minor)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR[%i]: cudaGetDeviceCount() return %s\n",
                cudaGetErrorString(err));
        return;
    }


    if (deviceId > device_count - 1)
    {
        fprintf(stderr, "ERROR: Device ID out of range!\n");
        return;
    }

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, deviceId);
    *major = device_prop.major;
    *minor = device_prop.minor;
}


bool oskar_cuda_device_supports_double(const int deviceId)
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR[%i]: cudaGetDeviceCount() return %s\n",
                cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, deviceId);

    int major = device_prop.major;
    int minor = device_prop.minor;

    if (major >= 2)
        return true;
    else if (minor >= 3)
        return true;
    else
        return false;
}


#ifdef __cplusplus
}
#endif


