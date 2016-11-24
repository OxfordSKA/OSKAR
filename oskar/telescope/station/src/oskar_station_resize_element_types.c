/*
 * Copyright (c) 2013-2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
    int i, old_num_element_types;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the old size. */
    old_num_element_types = model->num_element_types;

    /* Check if increasing or decreasing in size. */
    if (num_element_types > old_num_element_types)
    {
        /* Enlarge the element array and create new elements. */
        model->element = realloc(model->element,
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
        model->element = realloc(model->element,
                num_element_types * sizeof(oskar_Element*));
    }

    /* Update the new size. */
    model->num_element_types = num_element_types;
}

#ifdef __cplusplus
}
#endif
