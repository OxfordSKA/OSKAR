/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_append_to_set(int* set_size, oskar_Sky*** set_ptr,
        int max_sources_per_model, const oskar_Sky* model, int* status)
{
    int free_space = 0, space_required = 0;
    int num_extra_models = 0, number_to_copy = 0;
    int i = 0, j = 0, from_offset = 0;
    oskar_Sky **set = 0;
    size_t new_size = 0;
    if (*status) return;

    /* Get type and location. */
    const int type     = oskar_sky_precision(model);
    const int location = oskar_sky_mem_location(model);
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Work out if the set needs to be resized, and if so by how much. */
    free_space = (*set_size == 0) ? 0 :
            max_sources_per_model - (*set_ptr)[*set_size - 1]->num_sources;
    space_required = model->num_sources - free_space;
    num_extra_models = (space_required + max_sources_per_model - 1)
            / max_sources_per_model;

    /* Sanity check. */
    if (num_extra_models < 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Resize the array of sky model handles. */
    new_size = (*set_size + num_extra_models) * sizeof(oskar_Sky*);
    *set_ptr = (oskar_Sky**) realloc((*set_ptr), new_size);
    set = *set_ptr;
    for (i = *set_size; i < *set_size + num_extra_models; ++i)
    {
        set[i] = oskar_sky_create(type, location,
                max_sources_per_model, status);
        /* TODO Please clarify this statement and explain why it's needed. */
        /* Set the number sources to zero as this is the number currently
         * allocated in the model. */
        set[i]->num_sources = 0;
    }

    if (*status) return;

    /* Copy sources from the model into the set. */
    /* Loop over set entries with free space and copy sources into them. */
    number_to_copy = model->num_sources;
    from_offset = 0;
    for (i = (*set_size-1 > 0) ? *set_size-1 : 0;
            i < *set_size + num_extra_models; ++i)
    {
        int n_copy = 0, offset_dst = 0;
        offset_dst = oskar_sky_num_sources(set[i]);
        free_space = max_sources_per_model - offset_dst;
        n_copy = MIN(free_space, number_to_copy);
        oskar_sky_copy_contents(set[i], model, offset_dst, from_offset,
                n_copy, status);
        if (*status) break;
        set[i]->num_sources = n_copy + offset_dst;
        number_to_copy     -= n_copy;
        from_offset        += n_copy;
    }

    if (*status) return;

    /* Set the use extended flag if needed. */
    for (j = (*set_size-1 > 0) ? *set_size-1 : 0;
            j < *set_size + num_extra_models; ++j)
    {
        oskar_Sky* sky = set[j];
        const oskar_Mem *major = 0, *minor = 0;

        /* If any source in the model is extended, set the flag. */
        major = oskar_sky_fwhm_major_rad_const(sky);
        minor = oskar_sky_fwhm_minor_rad_const(sky);
        if (type == OSKAR_DOUBLE)
        {
            const double *maj_ = 0, *min_ = 0;
            maj_ = oskar_mem_double_const(major, status);
            min_ = oskar_mem_double_const(minor, status);
            for (i = 0; i < sky->num_sources; ++i)
            {
                if (maj_[i] > 0.0 || min_[i] > 0.0)
                {
                    sky->use_extended = OSKAR_TRUE;
                    break;
                }
            }
        }
        else
        {
            const float *maj_ = 0, *min_ = 0;
            maj_ = oskar_mem_float_const(major, status);
            min_ = oskar_mem_float_const(minor, status);
            for (i = 0; i < sky->num_sources; ++i)
            {
                if (maj_[i] > 0.0 || min_[i] > 0.0)
                {
                    sky->use_extended = OSKAR_TRUE;
                    break;
                }
            }
        }
    }

    /* Update the set size */
    *set_size += num_extra_models;
}

#ifdef __cplusplus
}
#endif
