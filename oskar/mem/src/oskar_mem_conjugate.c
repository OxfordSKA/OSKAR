/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "mem/define_mem_conjugate.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_MEM_CONJ(mem_conj_float, float2)
OSKAR_MEM_CONJ(mem_conj_double, double2)

void oskar_mem_conjugate(oskar_Mem* mem, int* status)
{
    const int location = oskar_mem_location(mem);
    const unsigned int n = (unsigned int) oskar_mem_length(mem);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (oskar_mem_type(mem) == OSKAR_SINGLE_COMPLEX)
        {
            mem_conj_float(n, oskar_mem_float2(mem, status));
        }
        else if (oskar_mem_type(mem) == OSKAR_DOUBLE_COMPLEX)
        {
            mem_conj_double(n, oskar_mem_double2(mem, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        const char* k = 0;
        if (oskar_mem_type(mem) == OSKAR_SINGLE_COMPLEX)
        {
            k = "mem_conj_float";
        }
        else if (oskar_mem_type(mem) == OSKAR_DOUBLE_COMPLEX)
        {
            k = "mem_conj_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        if (!*status)
        {
            size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(
                    oskar_mem_length(mem), local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &n},
                    {PTR_SZ, oskar_mem_buffer(mem)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
