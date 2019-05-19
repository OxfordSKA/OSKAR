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
            fftphase_float(num_x, num_y,
                    oskar_mem_float(complex_data, status));
        else if (type == OSKAR_DOUBLE_COMPLEX)
            fftphase_double(num_x, num_y,
                    oskar_mem_double(complex_data, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {32, 8, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE_COMPLEX)      k = "fftphase_float";
        else if (type == OSKAR_DOUBLE_COMPLEX) k = "fftphase_double";
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
