/*
 * Copyright (c) 2018-2019, The University of Oxford
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

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "utility/oskar_device_count.h"

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_device_count(const char* platform, int* location)
{
    int i, num_cuda = 0, num_cl = 0;
    char selector = ' ';
    const char* env = getenv("OSKAR_PLATFORM");
    if (platform && strlen(platform) > 0) selector = toupper(platform[0]);
    else if (env && strlen(env) > 0) selector = toupper(env[0]);
#ifdef OSKAR_HAVE_CUDA
    if (cudaGetDeviceCount(&num_cuda) != cudaSuccess) num_cuda = 0;
#endif
    if ((selector == ' ' && num_cuda > 0) || selector == 'C')
    {
        if (location) *location = OSKAR_GPU;
        return num_cuda;
    }
    /* Only need this for num_cl. */
    oskar_Device** devices = oskar_device_create_list(OSKAR_CL, &num_cl);
    for (i = 0; i < num_cl; ++i) oskar_device_free(devices[i]);
    free(devices);
    if ((selector == ' ' && num_cl > 0) || selector == 'O')
    {
        if (location) *location = OSKAR_CL;
        return num_cl;
    }
    if (location) *location = OSKAR_CPU;
    return 0;
}

#ifdef __cplusplus
}
#endif
