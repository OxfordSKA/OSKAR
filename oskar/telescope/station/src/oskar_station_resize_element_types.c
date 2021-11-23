/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_resize_element_types(oskar_Station* model,
        int num_element_types, int* status)
{
    int i = 0, old_num_element_types = 0;
    if (*status || !model) return;

    /* Get the old size. */
    old_num_element_types = model->num_element_types;

    /* Check if increasing or decreasing in size. */
    if (num_element_types > old_num_element_types)
    {
        /* Enlarge the element array and create new elements. */
        model->element = (oskar_Element**) realloc(model->element,
                num_element_types * sizeof(oskar_Element*));
        if (!model->element)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }
        for (i = old_num_element_types; i < num_element_types; ++i)
        {
            model->element[i] = oskar_element_create(
                    oskar_station_precision(model),
                    oskar_station_mem_location(model), status);
        }
    }
    else if (num_element_types < old_num_element_types)
    {
        /* Free old elements and shrink the element array. */
        for (i = num_element_types; i < old_num_element_types; ++i)
        {
            oskar_element_free(oskar_station_element(model, i), status);
        }
        model->element = (oskar_Element**) realloc(model->element,
                num_element_types * sizeof(oskar_Element*));
    }

    /* Update the new size. */
    model->num_element_types = num_element_types;
}

#ifdef __cplusplus
}
#endif
