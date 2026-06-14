/*
 * Copyright (c) 2013-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


double oskar_sky_double(const oskar_Sky* sky, oskar_SkyAttribDouble attribute)
{
    return (attribute >= OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE ? 0. :
            sky->attr_double[attribute]
    );
}


int oskar_sky_int(const oskar_Sky* sky, oskar_SkyAttribInt attribute)
{
    return (attribute >= OSKAR_SKY_NUM_ATTRIBUTES_INT ? 0 :
            sky->attr_int[attribute]
    );
}


void oskar_sky_set_double(
        oskar_Sky* sky,
        oskar_SkyAttribDouble attribute,
        double value
)
{
    switch (attribute)
    {
    case OSKAR_SKY_REF_RA_RAD:
    case OSKAR_SKY_REF_DEC_RAD:
        sky->attr_double[attribute] = value;
        break;
    default:                                              /* LCOV_EXCL_LINE */
        break; /* Unreachable. */                         /* LCOV_EXCL_LINE */
    }
}


void oskar_sky_set_int(oskar_Sky* sky, oskar_SkyAttribInt attribute, int value)
{
    switch (attribute)
    {
    case OSKAR_SKY_PRECISION:
    case OSKAR_SKY_MEM_LOCATION:
    case OSKAR_SKY_CAPACITY:
    case OSKAR_SKY_NUM_COLUMNS:
        /* Read-only attributes - do nothing, as these can't be set here. */
        break;
    case OSKAR_SKY_NUM_SOURCES:
        /* The number of (valid) sources can be set,
         * as long as it's less than the capacity. */
        if (value < sky->attr_int[OSKAR_SKY_CAPACITY])
        {
            sky->attr_int[attribute] = value;
        }
        break;
    case OSKAR_SKY_USE_EXTENDED:
        sky->attr_int[attribute] = value;
        break;
    default:                                              /* LCOV_EXCL_LINE */
        break; /* Unreachable. */                         /* LCOV_EXCL_LINE */
    }
}


int oskar_sky_first_column(const oskar_Sky* sky, oskar_SkyColumn column_type)
{
    int i = 0;
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (; i < num_columns; ++i)
    {
        if (sky->column_type[i] == column_type) return i;
    }
    return -1;
}


int oskar_sky_num_columns_of_type(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type
)
{
    int i = 0;
    int count = 0;
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    for (; i < num_columns; ++i)
    {
        if (sky->column_type[i] == column_type) count++;
    }
    return count;
}


int oskar_sky_num_valid_columns_of_type(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int index
)
{
    int num = 0, status = 0;
    index = index * (int) OSKAR_SKY_NUM_FIXED_COLUMN_TYPES + (int) column_type;
    if (sky->attr_int[OSKAR_SKY_MEM_LOCATION] == OSKAR_CPU)
    {
        const int* ptr = (const int*) oskar_mem_void_const(
                sky->num_valid_columns
        );
        return ptr[index];
    }
    oskar_mem_read_element(sky->num_valid_columns, index, &num, &status);
    return num;
}


void oskar_sky_set_num_valid_columns_of_type(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int index,
        int value
)
{
    index = index * (int) OSKAR_SKY_NUM_FIXED_COLUMN_TYPES + (int) column_type;
    if (sky->attr_int[OSKAR_SKY_MEM_LOCATION] == OSKAR_CPU)
    {
        ((int*) oskar_mem_void(sky->num_valid_columns))[index] = value;
    }
}


int oskar_sky_column_attribute(const oskar_Sky* sky, int column_index)
{
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    return (column_index < 0 || column_index >= num_columns ?
            -1 : sky->column_attr[column_index]
    );
}


oskar_SkyColumn oskar_sky_column_type(const oskar_Sky* sky, int column_index)
{
    const int num_columns = sky->attr_int[OSKAR_SKY_NUM_COLUMNS];
    return (column_index < 0 || column_index >= num_columns ?
            OSKAR_SKY_CUSTOM : (oskar_SkyColumn) sky->column_type[column_index]
    );
}


void oskar_sky_set_data(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int index,
        double value,
        int* status
)
{
    oskar_Mem* column = oskar_sky_column(
            sky, column_type, column_attribute, status
    );
    if (column)
    {
        oskar_mem_set_element_real(column, index, value, status);

        /* Update the number of valid columns of this type for this source,
         * if required. */
        if ((int) column_type < (int) OSKAR_SKY_NUM_FIXED_COLUMN_TYPES)
        {
            const int num_valid = oskar_sky_num_valid_columns_of_type(
                    sky, column_type, index
            );
            if (num_valid < column_attribute + 1)
            {
                oskar_sky_set_num_valid_columns_of_type(
                        sky, column_type, index, column_attribute + 1
                );
            }
        }
    }
}


double oskar_sky_data(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int index
)
{
    int status = 0;
    const oskar_Mem* column = oskar_sky_column_const(
            sky, column_type, column_attribute
    );
    return column ? oskar_mem_get_element(column, index, &status) : 0.0;
}

#ifdef __cplusplus
}
#endif
