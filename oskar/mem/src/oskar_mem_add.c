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
#include "utility/oskar_device.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add(
        oskar_Mem* out,
        const oskar_Mem* in1,
        const oskar_Mem* in2,
        size_t offset_out,
        size_t offset_in1,
        size_t offset_in2,
        size_t num_elements,
        int* status)
{
    oskar_Mem *a_temp = 0, *b_temp = 0;
    const oskar_Mem *a_, *b_; /* Pointers. */
    if (num_elements == 0) return;
    const int type = oskar_mem_type(out);
    const int precision = oskar_mem_precision(out);
    const int location = oskar_mem_location(out);
    if (*status) return;
    if (type != oskar_mem_type(in1) || type != oskar_mem_type(in2))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    a_ = in1;
    b_ = in2;
    if (oskar_mem_location(in1) != location)
    {
        a_temp = oskar_mem_create_copy(in1, location, status);
        a_ = a_temp;
    }
    if (oskar_mem_location(in2) != location)
    {
        b_temp = oskar_mem_create_copy(in2, location, status);
        b_ = b_temp;
    }
    if (oskar_mem_is_matrix(out))  { offset_out *= 4; num_elements *= 4; }
    if (oskar_mem_is_complex(out)) { offset_out *= 2; num_elements *= 2; }
    if (oskar_mem_is_matrix(in1))    offset_in1 *= 4;
    if (oskar_mem_is_complex(in1))   offset_in1 *= 2;
    if (oskar_mem_is_matrix(in2))    offset_in2 *= 4;
    if (oskar_mem_is_complex(in2))   offset_in2 *= 2;
    if (location == OSKAR_CPU)
    {
        size_t i;
        if (precision == OSKAR_DOUBLE)
        {
            double *c = oskar_mem_double(out, status);
            const double *a = oskar_mem_double_const(a_, status);
            const double *b = oskar_mem_double_const(b_, status);
            for (i = 0; i < num_elements; ++i)
                c[i + offset_out] = a[i + offset_in1] + b[i + offset_in2];
        }
        else if (precision == OSKAR_SINGLE)
        {
            float *c = oskar_mem_float(out, status);
            const float *a = oskar_mem_float_const(a_, status);
            const float *b = oskar_mem_float_const(b_, status);
            for (i = 0; i < num_elements; ++i)
                c[i + offset_out] = a[i + offset_in1] + b[i + offset_in2];
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const unsigned int off_a = (unsigned int) offset_in1;
        const unsigned int off_b = (unsigned int) offset_in2;
        const unsigned int off_c = (unsigned int) offset_out;
        const unsigned int n = (unsigned int) num_elements;
        const char* k = 0;
        if (precision == OSKAR_DOUBLE)      k = "mem_add_double";
        else if (precision == OSKAR_SINGLE) k = "mem_add_float";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off_a},
                {INT_SZ, &off_b},
                {INT_SZ, &off_c},
                {INT_SZ, &n},
                {PTR_SZ, oskar_mem_buffer_const(a_)},
                {PTR_SZ, oskar_mem_buffer_const(b_)},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }

    /* Free temporary arrays. */
    oskar_mem_free(a_temp, status);
    oskar_mem_free(b_temp, status);
}

#ifdef __cplusplus
}
#endif
