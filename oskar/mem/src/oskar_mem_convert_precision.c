/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_mem_convert_precision(const oskar_Mem* input,
        int output_precision, int* status)
{
    oskar_Mem *output = 0, *in_temp = 0;
    const oskar_Mem *in = 0;
    int input_precision, type;
    size_t num_elements, i;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* If input and output precision are the same,
     * return a carbon copy of the input data. */
    input_precision = oskar_mem_precision(input);
    if (input_precision == output_precision)
    {
        return oskar_mem_create_copy(input, OSKAR_CPU, status);
    }

    /* Copy source data to CPU memory if necessary. */
    if (oskar_mem_location(input) != OSKAR_CPU)
    {
        in_temp = oskar_mem_create_copy(input, OSKAR_CPU, status);
        in = in_temp;
    }
    else
    {
        in = input;
    }

    /* Create a new array to hold the converted data. */
    type = output_precision;
    num_elements = oskar_mem_length(in);
    if (oskar_mem_is_complex(in))
    {
        type |= OSKAR_COMPLEX;
        num_elements *= 2;
    }
    if (oskar_mem_is_matrix(in))
    {
        type |= OSKAR_MATRIX;
        num_elements *= 4;
    }
    output = oskar_mem_create(type, OSKAR_CPU, oskar_mem_length(in), status);

    /* Convert the data. */
    if (input_precision == OSKAR_SINGLE &&
            output_precision == OSKAR_DOUBLE)
    {
        const float* src_;
        double* dst_;
        src_ = oskar_mem_float_const(in, status);
        dst_ = oskar_mem_double(output, status);
        for (i = 0; i < num_elements; ++i)
        {
            dst_[i] = src_[i];
        }
    }
    else if (input_precision == OSKAR_DOUBLE &&
            output_precision == OSKAR_SINGLE)
    {
        const double* src_;
        float* dst_;
        src_ = oskar_mem_double_const(in, status);
        dst_ = oskar_mem_float(output, status);
        for (i = 0; i < num_elements; ++i)
        {
            dst_[i] = src_[i];
        }
    }
    else
    {
        oskar_mem_free(output, status);
        output = 0;
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    oskar_mem_free(in_temp, status);
    return output;
}

#ifdef __cplusplus
}
#endif
