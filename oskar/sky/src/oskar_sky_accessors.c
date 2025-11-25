/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
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
    case OSKAR_SKY_NUM_SOURCES:
    case OSKAR_SKY_NUM_COLUMNS:
        /* Read-only attributes - do nothing, as these can't be set here. */
        break;
    case OSKAR_SKY_USE_EXTENDED:
        sky->attr_int[attribute] = value;
        break;
    default:                                              /* LCOV_EXCL_LINE */
        break; /* Unreachable. */                         /* LCOV_EXCL_LINE */
    }
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
    if (column) oskar_mem_set_element_real(column, index, value, status);
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
