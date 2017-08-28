/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "utility/oskar_device_utils.h"
#include <stdlib.h>
#include <string.h>

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_device_check_error(int* status)
{
    if (*status) return;

#ifdef OSKAR_HAVE_CUDA
    *status = (int) cudaPeekAtLastError();
#endif
}


int oskar_device_count(int *status)
{
    int num = 0;
    if (*status) return 0;
#ifdef OSKAR_HAVE_CUDA
    if (cudaGetDeviceCount(&num) != cudaSuccess)
        num = 0;
#endif
    return num;
}


void oskar_device_mem_info(size_t* mem_free, size_t* mem_total)
{
    if (!mem_free || !mem_total) return;
#ifdef OSKAR_HAVE_CUDA
    cudaMemGetInfo(mem_free, mem_total);
#else
    (void) mem_free;
    (void) mem_total;
#endif
}


char* oskar_device_name(int device_id)
{
    char* name = 0;
#ifdef OSKAR_HAVE_CUDA
    struct cudaDeviceProp device_prop;
#endif
    if (device_id < 0) return 0;
#ifdef OSKAR_HAVE_CUDA
    cudaGetDeviceProperties(&device_prop, device_id);
    name = (char*) calloc(1 + strlen(device_prop.name), sizeof(char));
    strcpy(name, device_prop.name);
#endif
    return name;
}


void oskar_device_reset(void)
{
#ifdef OSKAR_HAVE_CUDA
    cudaDeviceReset();
#endif
}


void oskar_device_set(int id, int* status)
{
    if (*status || id < 0) return;

#ifdef OSKAR_HAVE_CUDA
    *status = (int) cudaSetDevice(id);
#endif
}


void oskar_device_synchronize(void)
{
#ifdef OSKAR_HAVE_CUDA
    cudaDeviceSynchronize();
#endif
}


#ifdef __cplusplus
}
#endif
