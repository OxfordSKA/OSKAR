/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/define_grid_correction.h"
#include "imager/oskar_grid_correction.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_GRID_CORRECTION(grid_correction_float, float)
OSKAR_GRID_CORRECTION(grid_correction_double, double)

void oskar_grid_correction(const int image_size,
        const oskar_Mem* corr_func, oskar_Mem* complex_image, int* status)
{
    oskar_Mem* corr_func_copy = 0;
    const oskar_Mem* corr_func_ptr = corr_func;
    const int location = oskar_mem_location(complex_image);
    const int type = oskar_mem_precision(complex_image);
    if (oskar_mem_location(corr_func) != location)
    {
        corr_func_copy = oskar_mem_create_copy(corr_func, location, status);
        corr_func_ptr = corr_func_copy;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            grid_correction_double(image_size,
                    oskar_mem_double_const(corr_func_ptr, status),
                    oskar_mem_double(complex_image, status));
        }
        else if (type == OSKAR_SINGLE)
        {
            grid_correction_float(image_size,
                    oskar_mem_float_const(corr_func_ptr, status),
                    oskar_mem_float(complex_image, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {16, 16, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE)
        {
            k = "grid_correction_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "grid_correction_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) image_size, local_size[0]);
        global_size[1] = oskar_device_global_size(
                (size_t) image_size, local_size[1]);
        const oskar_Arg args[] = {
                {INT_SZ, &image_size},
                {PTR_SZ, oskar_mem_buffer_const(corr_func_ptr)},
                {PTR_SZ, oskar_mem_buffer(complex_image)}
        };
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }

    /* Free the copy. */
    oskar_mem_free(corr_func_copy, status);
}

#ifdef __cplusplus
}
#endif
