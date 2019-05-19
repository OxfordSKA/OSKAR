/*
 * Copyright (c) 2016-2019, The University of Oxford
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
            grid_correction_double(image_size,
                    oskar_mem_double_const(corr_func_ptr, status),
                    oskar_mem_double(complex_image, status));
        else if (type == OSKAR_SINGLE)
            grid_correction_float(image_size,
                    oskar_mem_float_const(corr_func_ptr, status),
                    oskar_mem_float(complex_image, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {16, 16, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE)
            k = "grid_correction_float";
        else if (type == OSKAR_DOUBLE)
            k = "grid_correction_double";
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
                {PTR_SZ, oskar_mem_buffer_const(corr_func)},
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
