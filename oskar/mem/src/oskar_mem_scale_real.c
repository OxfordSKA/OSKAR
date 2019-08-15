/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C"
#endif
void oskar_mem_scale_real(oskar_Mem* mem, double value,
        size_t offset, size_t num_elements, int* status)
{
    if (*status) return;
    const int location = mem->location;
    const int precision = oskar_type_precision(mem->type);
    if (oskar_type_is_complex(mem->type))
    {
        offset *= 2;
        num_elements *= 2;
    }
    if (oskar_type_is_matrix(mem->type))
    {
        offset *= 4;
        num_elements *= 4;
    }
    if (location == OSKAR_CPU)
    {
        size_t i;
        if (precision == OSKAR_SINGLE)
        {
            float *aa = ((float*) mem->data) + offset;
            const float value_f = (float) value;
            for (i = 0; i < num_elements; ++i) aa[i] *= value_f;
        }
        else if (precision == OSKAR_DOUBLE)
        {
            double *aa = ((double*) mem->data) + offset;
            for (i = 0; i < num_elements; ++i) aa[i] *= value;
        }
        else *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const unsigned int n = (unsigned int) num_elements;
        const unsigned int off = (unsigned int) offset;
        const float value_f = (float) value;
        const int is_dbl = (precision == OSKAR_DOUBLE);
        const char* k = 0;
        if (precision == OSKAR_SINGLE)      k = "mem_scale_float";
        else if (precision == OSKAR_DOUBLE) k = "mem_scale_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off},
                {INT_SZ, &n},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&value : (const void*)&value_f},
                {PTR_SZ, oskar_mem_buffer(mem)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}
