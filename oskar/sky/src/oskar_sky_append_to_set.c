/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifdef __cplusplus
extern "C" {
#endif


void oskar_sky_append_to_set(
        int* set_size,
        oskar_Sky*** set_ptr,
        int max_sources_per_model,
        const oskar_Sky* model,
        int* status
)
{
    int free_space = 0;
    int number_to_copy = 0;
    int i = 0, j = 0, from_offset = 0;
    oskar_Sky **set = 0;
    if (*status) return;

    /* Get type and location. */
    const int type = oskar_sky_int(model, OSKAR_SKY_PRECISION);
    const int location = oskar_sky_int(model, OSKAR_SKY_MEM_LOCATION);
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Work out if the set needs to be resized, and if so by how much. */
    const int num_sources_in = oskar_sky_int(model, OSKAR_SKY_NUM_SOURCES);
    const int num_sources_last = ((*set_size == 0) ? 0 :
            oskar_sky_int((*set_ptr)[*set_size - 1], OSKAR_SKY_NUM_SOURCES)
    );
    free_space = (*set_size == 0) ? 0 :
            max_sources_per_model - num_sources_last;
    const int space_required = num_sources_in - free_space;
    const int num_extra_models = (space_required + max_sources_per_model - 1)
            / max_sources_per_model;

    /* Sanity check. */
    if (num_extra_models < 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;             /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }

    /* Resize the array of sky model handles. */
    const size_t new_size = *set_size + num_extra_models;
    *set_ptr = (oskar_Sky**) realloc(*set_ptr, new_size * sizeof(oskar_Sky*));
    set = *set_ptr;
    for (i = *set_size; i < *set_size + num_extra_models; ++i)
    {
        set[i] = oskar_sky_create(
                type, location, max_sources_per_model, status
        );
        /* Set the number sources to zero as this is the number currently
         * allocated in the model. */
        set[i]->attr_int[OSKAR_SKY_NUM_SOURCES] = 0;
    }

    if (*status) return;

    /* Copy sources from the model into the set. */
    /* Loop over set entries with free space and copy sources into them. */
    number_to_copy = num_sources_in;
    from_offset = 0;
    for (i = (*set_size - 1 > 0) ? *set_size - 1 : 0;
            i < *set_size + num_extra_models; ++i)
    {
        const int offset_dst = oskar_sky_int(set[i], OSKAR_SKY_NUM_SOURCES);
        free_space = max_sources_per_model - offset_dst;
        const int n_copy = MIN(free_space, number_to_copy);
        oskar_sky_copy_contents(
                set[i], model, offset_dst, from_offset, n_copy, status
        );
        if (*status) break;
        set[i]->attr_int[OSKAR_SKY_NUM_SOURCES] = n_copy + offset_dst;
        number_to_copy -= n_copy;
        from_offset    += n_copy;
    }

    if (*status) return;

    /* Set the use extended flag if needed. */
    for (j = (*set_size - 1 > 0) ? *set_size - 1 : 0;
            j < *set_size + num_extra_models; ++j)
    {
        oskar_Sky* sky = set[j];
        const oskar_Mem *major = 0, *minor = 0;

        /* If any source in the model is extended, set the flag. */
        major = oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0);
        minor = oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0);
        if (!major || !minor) continue;
        const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
        if (type == OSKAR_DOUBLE)
        {
            const double *maj_ = 0, *min_ = 0;
            maj_ = oskar_mem_double_const(major, status);
            min_ = oskar_mem_double_const(minor, status);
            for (i = 0; i < num_sources; ++i)
            {
                if (maj_[i] > 0.0 || min_[i] > 0.0)
                {
                    oskar_sky_set_int(sky, OSKAR_SKY_USE_EXTENDED, 1);
                    break;
                }
            }
        }
        else
        {
            const float *maj_ = 0, *min_ = 0;
            maj_ = oskar_mem_float_const(major, status);
            min_ = oskar_mem_float_const(minor, status);
            for (i = 0; i < num_sources; ++i)
            {
                if (maj_[i] > 0.0 || min_[i] > 0.0)
                {
                    oskar_sky_set_int(sky, OSKAR_SKY_USE_EXTENDED, 1);
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
