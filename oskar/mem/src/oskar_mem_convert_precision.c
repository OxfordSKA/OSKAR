/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int input_precision = 0, type = 0;
    size_t num_elements = 0, i = 0;
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
        const float* src_ = 0;
        double* dst_ = 0;
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
        const double* src_ = 0;
        float* dst_ = 0;
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
