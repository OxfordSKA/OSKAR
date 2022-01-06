/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SDF "% .6e "
#define SDD "% .14e "

void oskar_mem_save_ascii(FILE* file, size_t num_mem,
        size_t offset, size_t num_elements, int* status, ...)
{
    size_t i = 0, j = 0;
    va_list args;
    oskar_Mem** handles = 0; /* Array of oskar_Mem pointers in CPU memory. */
    if (*status) return;

    /* Check for at least one array. */
    if (num_mem == 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check there are at least the number of specified elements in
     * each array. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem = va_arg(args, const oskar_Mem*);
        if (oskar_mem_length(mem) < num_elements)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
        }
    }
    va_end(args);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Allocate and set up the handle array. */
    handles = (oskar_Mem**) calloc(num_mem, sizeof(oskar_Mem*));
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        oskar_Mem* mem = va_arg(args, oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            handles[i] = oskar_mem_create_copy(mem, OSKAR_CPU, status);
        }
        else
        {
            handles[i] = mem;
        }
    }
    va_end(args);

    for (j = 0; j < num_elements; ++j)
    {
        /* Break if error. */
        if (*status) break;

        for (i = 0; i < num_mem; ++i)
        {
            const void* data = oskar_mem_void_const(handles[i]);
            switch (oskar_mem_type(handles[i]))
            {
            case OSKAR_SINGLE:
            {
                fprintf(file, SDF, ((const float*)data)[j + offset]);
                continue;
            }
            case OSKAR_DOUBLE:
            {
                fprintf(file, SDD, ((const double*)data)[j + offset]);
                continue;
            }
            case OSKAR_SINGLE_COMPLEX:
            {
                float2 d;
                d = ((const float2*)data)[j + offset];
                fprintf(file, SDF SDF, d.x, d.y);
                continue;
            }
            case OSKAR_DOUBLE_COMPLEX:
            {
                double2 d;
                d = ((const double2*)data)[j + offset];
                fprintf(file, SDD SDD, d.x, d.y);
                continue;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                float4c d;
                d = ((const float4c*)data)[j + offset];
                fprintf(file, SDF SDF SDF SDF SDF SDF SDF SDF,
                        d.a.x, d.a.y, d.b.x, d.b.y, d.c.x, d.c.y, d.d.x, d.d.y);
                continue;
            }
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                double4c d;
                d = ((const double4c*)data)[j + offset];
                fprintf(file, SDD SDD SDD SDD SDD SDD SDD SDD,
                        d.a.x, d.a.y, d.b.x, d.b.y, d.c.x, d.c.y, d.d.x, d.d.y);
                continue;
            }
            case OSKAR_CHAR:
            {
                putc(((const char*)data)[j + offset], file);
                continue;
            }
            case OSKAR_INT:
            {
                fprintf(file, "%5d ", ((const int*)data)[j + offset]);
                continue;
            }
            default:
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                continue;
            }
            }
        }
        putc('\n', file);
    }

    /* Free any temporary memory used by this function. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem = 0;
        mem = va_arg(args, const oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            oskar_mem_free(handles[i], status);
        }
    }
    va_end(args);

    /* Free the handle array. */
    free(handles);
}

#ifdef __cplusplus
}
#endif
