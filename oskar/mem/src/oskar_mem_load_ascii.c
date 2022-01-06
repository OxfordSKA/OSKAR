/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/oskar_mem_load_ascii.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static void store_data(void* data, int type, size_t r, size_t* c,
        const double* row_data, int* status);

static void set_up_handles_and_defaults(size_t num_mem, oskar_Mem** mem_handle,
        double** row_defaults, size_t* num_cols_min, size_t* num_cols_max,
        va_list args, int* status);

size_t oskar_mem_load_ascii(const char* filename, size_t num_mem,
        int* status, ...)
{
    size_t i = 0;               /* Loop counter. */
    size_t row_index = 0;       /* Current row index loaded from file. */
    size_t buffer_size = 0;     /* Line buffer size. */
    size_t num_cols_min = 0;    /* Minimum number of columns required. */
    size_t num_cols_max = 0;    /* Maximum number of columns required. */
    FILE* file = 0;             /* File handle. */
    oskar_Mem** mem_handle = 0; /* Array of oskar_Mem handles in CPU memory. */
    double* row_data = 0;       /* Array to hold data from one row of file. */
    double* row_defaults = 0;   /* Array to hold default data for one row. */
    char* line = 0;             /* Line buffer. */
    va_list args;               /* Variable argument list. */

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check for at least one array. */
    if (num_mem == 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return 0;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Allocate array of handles to data in CPU memory. */
    mem_handle = (oskar_Mem**) calloc(num_mem, sizeof(oskar_Mem*));
    if (!mem_handle)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        goto cleanup;
    }

    /* Set up handles and default row data, and find the minimum and maximum
     * number of columns. */
    va_start(args, status);
    set_up_handles_and_defaults(num_mem, mem_handle, &row_defaults,
            &num_cols_min, &num_cols_max, args, status);
    va_end(args);

    /* Allocate an array to hold numeric data for one row of the file. */
    if (*status) goto cleanup;
    row_data = (double*) calloc(num_cols_max, sizeof(double));
    if (!row_data)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        goto cleanup;
    }

    /* Loop over lines in the file. */
    while (oskar_getline(&line, &buffer_size, file) >= 0)
    {
        size_t num_cols_read = 0, col_index = 0;

        /* Break if error. */
        if (*status) break;

        /* Get the row's data from the file, skipping the row if there aren't
         * enough columns to read. */
        num_cols_read = oskar_string_to_array_d(line, num_cols_max, row_data);
        if (num_cols_read < num_cols_min) continue;

        /* Copy defaults to fill out any missing row data as needed. */
        for (i = num_cols_read; i < num_cols_max; ++i)
        {
            row_data[i] = row_defaults[i];
        }

        /* Loop over arrays passed to this function. */
        for (i = 0; i < num_mem; ++i)
        {
            /* Resize the array if it isn't big enough to hold the new data. */
            if (oskar_mem_length(mem_handle[i]) <= row_index)
            {
                oskar_mem_realloc(mem_handle[i], row_index + 1000, status);
                if (*status) break;
            }

            /* Store data. */
            store_data(oskar_mem_void(mem_handle[i]),
                    oskar_mem_type(mem_handle[i]), row_index, &col_index,
                    row_data, status);
        }

        /* Increment row counter. */
        ++row_index;
    }

    /* Ensure that the handle array exists. */
cleanup:
    if (mem_handle)
    {
        /* Resize all arrays to their actual final length. */
        for (i = 0; i < num_mem; ++i)
        {
            oskar_mem_realloc(mem_handle[i], row_index, status);
        }

        /* Free any temporary memory used by this function. */
        va_start(args, status);
        for (i = 0; i < num_mem; ++i)
        {
            oskar_Mem* mem = va_arg(args, oskar_Mem*);
            if (oskar_mem_location(mem) != OSKAR_CPU)
            {
                oskar_mem_copy(mem, mem_handle[i], status);
                oskar_mem_free(mem_handle[i], status);
            }

            /* We don't need the default here, but must advance the va_arg
             * pointer to get to the next array. */
            (void)va_arg(args, const char*);
        }
        va_end(args);
    }

    /* Free local arrays. */
    free(line);
    free(row_data);
    free(row_defaults);
    free(mem_handle);
    fclose(file);
    return *status ? 0 : row_index;
}


static void set_up_handles_and_defaults(size_t num_mem, oskar_Mem** mem_handle,
        double** row_defaults, size_t* num_cols_min, size_t* num_cols_max,
        va_list args, int* status)
{
    size_t i = 0, buffer_size = 0, col_start = 0;
    char* line = 0;

    /* Loop over arrays passed to this function to set up handles and
     * default row data. */
    for (i = 0; i < num_mem; ++i)
    {
        oskar_Mem* mem = 0;
        size_t num_cols_needed = 1;

        /* Break if error. */
        if (*status) break;

        /* Get and store a handle to CPU-accessible memory for the array. */
        mem = va_arg(args, oskar_Mem*);
        mem_handle[i] = mem;
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            mem_handle[i] = oskar_mem_create(oskar_mem_type(mem),
                    OSKAR_CPU, oskar_mem_length(mem), status);
        }

        /* Break if error. */
        if (*status) break;

        /* Determine number of columns needed for this array, and increment
         * the maximum column count. */
        if (oskar_mem_is_complex(mem_handle[i])) num_cols_needed *= 2;
        if (oskar_mem_is_matrix(mem_handle[i]))  num_cols_needed *= 4;
        *num_cols_max += num_cols_needed;

        /* Resize array to hold row defaults. */
        *row_defaults = (double*) realloc(*row_defaults,
                *num_cols_max * sizeof(double));
        if (!*row_defaults)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            break;
        }

        /* Make a copy of the default string from the argument list. */
        const char* def = va_arg(args, const char*);
        const size_t def_len = 1 + strlen(def);
        if (buffer_size < def_len)
        {
            buffer_size = def_len;
            line = (char*) realloc(line, buffer_size);
            if (!line)
            {
                *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
                break;
            }
        }
        memcpy(line, def, def_len);

        /* Get default value(s) for the array, and store them at the
         * current column start. */
        const size_t num_defaults = oskar_string_to_array_d(
                line, num_cols_needed, *row_defaults + col_start);
        if (num_defaults == 0)
        {
            *num_cols_min += num_cols_needed;
            if (*num_cols_min != *num_cols_max)
            {
                /* A default value was set for one or more earlier columns,
                 * which makes no sense. */
                *status = OSKAR_ERR_DIMENSION_MISMATCH;
            }
        }
        else if (num_defaults != num_cols_needed)
        {
            /* The number of defaults provided does not correspond to the
             * number required for the data type. */
            *status = OSKAR_ERR_TYPE_MISMATCH;
        }
        col_start += num_cols_needed;
    }
    free(line);
}


static void store_data(void* data, int type, size_t r, size_t* c,
        const double* row_data, int* status)
{
    /* Store the new data for the current row and column. */
    switch (type)
    {
    case OSKAR_SINGLE:
    {
        ((float*)data)[r] = (float) row_data[(*c)++];
        return;
    }
    case OSKAR_DOUBLE:
    {
        ((double*)data)[r] = row_data[(*c)++];
        return;
    }
    case OSKAR_SINGLE_COMPLEX:
    {
        float2* d = (float2*)data + r;
        d->x = (float) row_data[(*c)++];
        d->y = (float) row_data[(*c)++];
        return;
    }
    case OSKAR_DOUBLE_COMPLEX:
    {
        double2* d = (double2*)data + r;
        d->x = row_data[(*c)++];
        d->y = row_data[(*c)++];
        return;
    }
    case OSKAR_SINGLE_COMPLEX_MATRIX:
    {
        float4c* d = (float4c*)data + r;
        d->a.x = (float) row_data[(*c)++];
        d->a.y = (float) row_data[(*c)++];
        d->b.x = (float) row_data[(*c)++];
        d->b.y = (float) row_data[(*c)++];
        d->c.x = (float) row_data[(*c)++];
        d->c.y = (float) row_data[(*c)++];
        d->d.x = (float) row_data[(*c)++];
        d->d.y = (float) row_data[(*c)++];
        return;
    }
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
    {
        double4c* d = (double4c*)data + r;
        d->a.x = row_data[(*c)++];
        d->a.y = row_data[(*c)++];
        d->b.x = row_data[(*c)++];
        d->b.y = row_data[(*c)++];
        d->c.x = row_data[(*c)++];
        d->c.y = row_data[(*c)++];
        d->d.x = row_data[(*c)++];
        d->d.y = row_data[(*c)++];
        return;
    }
    case OSKAR_INT:
    {
        ((int*)data)[r] = (int) floor(row_data[(*c)++] + 0.5);
        return;
    }
    default:
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    }
}

#ifdef __cplusplus
}
#endif
