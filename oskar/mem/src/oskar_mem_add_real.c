/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add_real(oskar_Mem* mem, double val, int* status)
{
    size_t i = 0, num_elements = 0;
    if (*status) return;
    const int precision = oskar_mem_precision(mem);
    const int location = oskar_mem_location(mem);
    num_elements = oskar_mem_length(mem);
    if (num_elements == 0) return;
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (oskar_mem_is_matrix(mem)) num_elements *= 4;
    if (oskar_mem_is_complex(mem))
    {
        if (precision == OSKAR_DOUBLE)
        {
            double2 *t = oskar_mem_double2(mem, status);
            for (i = 0; i < num_elements; ++i) t[i].x += val;
        }
        else if (precision == OSKAR_SINGLE)
        {
            float2 *t = oskar_mem_float2(mem, status);
            for (i = 0; i < num_elements; ++i) t[i].x += val;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        if (precision == OSKAR_DOUBLE)
        {
            double *t = oskar_mem_double(mem, status);
            for (i = 0; i < num_elements; ++i) t[i] += val;
        }
        else if (precision == OSKAR_SINGLE)
        {
            float *t = oskar_mem_float(mem, status);
            for (i = 0; i < num_elements; ++i) t[i] += val;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
