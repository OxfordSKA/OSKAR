/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_copy_source_data(
        const oskar_Sky* in,
        const oskar_Mem* horizon_mask,
        const oskar_Mem* indices,
        oskar_Sky* out,
        int* status
)
{
    int c = 0, i = 0, num_out = 0;
    if (*status) return;
    const int location = oskar_sky_int(in, OSKAR_SKY_MEM_LOCATION);
    const int type = oskar_sky_int(out, OSKAR_SKY_PRECISION);
    const int num_in = oskar_sky_int(in, OSKAR_SKY_NUM_SOURCES);
    const int num_columns = oskar_sky_int(in, OSKAR_SKY_NUM_COLUMNS);
    const int capacity_in = oskar_sky_int(in, OSKAR_SKY_CAPACITY);
    const int capacity_out = oskar_sky_int(out, OSKAR_SKY_CAPACITY);
    for (c = 0; c < num_columns; ++c)
    {
        /* Touch each output column to make sure it exists, and
         * ensure that the columns in the output sky model are in
         * the same order as those in the input. */
        const oskar_SkyColumn column_type = oskar_sky_column_type(in, c);
        const int column_attribute = oskar_sky_column_attribute(in, c);
        (void) oskar_sky_column(out, column_type, column_attribute, status);
        if (column_type != oskar_sky_column_type(out, c) ||
                column_attribute != oskar_sky_column_attribute(out, c))
        {
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Sky models must have the same "   /* LCOV_EXCL_LINE */
                    "column order."                       /* LCOV_EXCL_LINE */
            );
            *status = OSKAR_ERR_TYPE_MISMATCH;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
    }
    if (location == OSKAR_CPU)
    {
        void* o_table = oskar_mem_void(out->table);
        const void* table = oskar_mem_void_const(in->table);
        const int* mask = oskar_mem_int_const(horizon_mask, status);
        (void) indices;
        if (type == OSKAR_SINGLE)
        {
            for (i = 0; i < num_in; ++i)
            {
                if (mask[i])
                {
                    #pragma GCC unroll 8
                    for (c = 0; c < num_columns; ++c)
                    {
                        ((float*) o_table)[c * capacity_out + num_out] =
                                ((const float*) table)[c * capacity_in + i];
                    }
                    num_out++;
                }
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            for (i = 0; i < num_in; ++i)
            {
                if (mask[i])
                {
                    #pragma GCC unroll 8
                    for (c = 0; c < num_columns; ++c)
                    {
                        ((double*) o_table)[c * capacity_out + num_out] =
                                ((const double*) table)[c * capacity_in + i];
                    }
                    num_out++;
                }
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_DOUBLE)
        {
            k = "copy_source_data_double";
        }
        else if (type == OSKAR_SINGLE)
        {
            k = "copy_source_data_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_in, local_size[0]
        );
        const oskar_Arg args[] = {
                {INT_SZ, &num_in},
                {INT_SZ, &capacity_in},
                {INT_SZ, &capacity_out},
                {PTR_SZ, oskar_mem_buffer_const(horizon_mask)},
                {PTR_SZ, oskar_mem_buffer_const(indices)},
                {INT_SZ, &num_columns},
                {PTR_SZ, oskar_mem_buffer_const(in->table)},
                {PTR_SZ, oskar_mem_buffer(out->table)}
        };
        oskar_device_launch_kernel(
                k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status
        );
        /* Last element of index array is the total number copied. */
        oskar_mem_read_element(indices, num_in, &num_out, status);
    }

    /* Copy metadata. */
    for (i = 0; i < OSKAR_SKY_NUM_ATTRIBUTES_INT; ++i)
    {
        const oskar_SkyAttribInt attr = (oskar_SkyAttribInt) i;
        oskar_sky_set_int(out, attr, oskar_sky_int(in, attr));
    }
    for (i = 0; i < OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE; ++i)
    {
        const oskar_SkyAttribDouble attr = (oskar_SkyAttribDouble) i;
        oskar_sky_set_double(out, attr, oskar_sky_double(in, attr));
    }

    /* Set the number of sources in the output sky model. */
    out->attr_int[OSKAR_SKY_NUM_SOURCES] = num_out;
}

#ifdef __cplusplus
}
#endif
