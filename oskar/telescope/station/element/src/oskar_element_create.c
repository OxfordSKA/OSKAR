/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

oskar_Element* oskar_element_create(int precision, int location, int* status)
{
    oskar_Element* data = (oskar_Element*) calloc(1, sizeof(oskar_Element));
    if (!data)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    data->precision = precision;
    data->mem_location = location;
    data->element_type = OSKAR_ELEMENT_TYPE_DIPOLE;
    data->taper_type = OSKAR_ELEMENT_TAPER_NONE;
    data->dipole_length = 0.5;
    data->dipole_length_units = OSKAR_WAVELENGTHS;
    if (precision != OSKAR_SINGLE && precision != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    return data;
}

#ifdef __cplusplus
}
#endif
