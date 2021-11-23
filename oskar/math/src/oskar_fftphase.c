/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_fftphase.h"
#include "math/oskar_fftphase.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_FFTPHASE(fftphase_float, float)
OSKAR_FFTPHASE(fftphase_double, double)

void oskar_fftphase(const int num_x, const int num_y,
        oskar_Mem* complex_data, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(complex_data);
    const int location = oskar_mem_location(complex_data);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE_COMPLEX)
        {
            fftphase_float(num_x, num_y,
                    oskar_mem_float(complex_data, status));
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            fftphase_double(num_x, num_y,
                    oskar_mem_double(complex_data, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {32, 8, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE_COMPLEX)
        {
            k = "fftphase_float";
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            k = "fftphase_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        if (num_x == 1)
        {
            local_size[0] = 1;
            local_size[1] = 256;
        }
        if (num_y == 1)
        {
            local_size[0] = 256;
            local_size[1] = 1;
        }
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        const oskar_Arg arg[] = {
                {INT_SZ, &num_x},
                {INT_SZ, &num_y},
                {PTR_SZ, oskar_mem_buffer(complex_data)}
        };
        global_size[0] = oskar_device_global_size(num_x, local_size[0]);
        global_size[1] = oskar_device_global_size(num_y, local_size[1]);
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(arg) / sizeof(oskar_Arg), arg, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
