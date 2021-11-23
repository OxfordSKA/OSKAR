/*
 * Copyright (c) 2018-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int i = 0, num_cuda = 0, num_cl = 0;
    char selector = ' ';
    const char* env = getenv("OSKAR_PLATFORM");
    if (platform && strlen(platform) > 0)
    {
        selector = toupper(platform[0]);
    }
    else if (env && strlen(env) > 0)
    {
        selector = toupper(env[0]);
    }
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
