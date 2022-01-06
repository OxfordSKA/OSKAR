/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_random_range(oskar_Mem* mem, double lo, double hi, int* status)
{
    oskar_Mem *temp = 0, *ptr = 0;
    size_t i = 0, num_elements = 0;
    int location = 0, precision = 0, type = 0;
    double r = 0.0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data and check type. */
    location = oskar_mem_location(mem);
    num_elements = oskar_mem_length(mem);
    type = oskar_mem_type(mem);
    precision = oskar_type_precision(type);
    if (precision != OSKAR_SINGLE && precision != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Initialise temporary memory array if required. */
    ptr = mem;
    if (location != OSKAR_CPU)
    {
        temp = oskar_mem_create(type, OSKAR_CPU, num_elements, status);
        if (*status)
        {
            oskar_mem_free(temp, status);
            return;
        }
        ptr = temp;
    }

    /* Get total number of elements. */
    if (oskar_type_is_matrix(type)) num_elements *= 4;
    if (oskar_type_is_complex(type)) num_elements *= 2;

    /* Fill memory with random numbers. */
    if (precision == OSKAR_SINGLE)
    {
        float *p = 0;
        p = oskar_mem_float(ptr, status);
        for (i = 0; i < num_elements; ++i)
        {
            /* NOLINTNEXTLINE: We can use rand() here without concern. */
            r = lo + (hi - lo) * (double)rand() / (double)RAND_MAX;
            p[i] = (float)r;
        }
    }
    else if (precision == OSKAR_DOUBLE)
    {
        double *p = 0;
        p = oskar_mem_double(ptr, status);
        for (i = 0; i < num_elements; ++i)
        {
            /* NOLINTNEXTLINE: We can use rand() here without concern. */
            r = lo + (hi - lo) * (double)rand() / (double)RAND_MAX;
            p[i] = r;
        }
    }

    /* Copy and clean up if required. */
    if (location != OSKAR_CPU)
    {
        oskar_mem_copy(mem, ptr, status);
    }
    oskar_mem_free(temp, status);
}

#ifdef __cplusplus
}
#endif
