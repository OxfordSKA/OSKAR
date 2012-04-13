/*
 * Copyright (c) 2011, The University of Oxford
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


#include "sky/oskar_sky_model_append_to_set.h"
#include "sky/oskar_sky_model_type.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_get_ptr.h"
#include "sky/oskar_sky_model_insert.h"
#include <stdio.h>
#include <math.h>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_append_to_set(int* set_size, oskar_SkyModel** set,
        int max_sources_per_model, const oskar_SkyModel* model)
{
    /* Declare variables */
    int free_space, space_required, num_extra_models, number_to_copy;
    int i, j, model_type, model_location;
    int n, n_copy, error, from_offset;
    oskar_SkyModel model_ptr;
    size_t new_size;

    if (set_size == NULL || set == NULL || model == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    model_type     = oskar_sky_model_type(model);
    model_location = oskar_sky_model_location(model);

    if (model_location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Work out if the set needs to be resized and if so by how much */
    free_space = (*set_size == 0) ?
            0 : max_sources_per_model - (*set)[*set_size - 1].num_sources;
    space_required = model->num_sources - free_space;
    num_extra_models = (int)ceil((double)space_required/(double)max_sources_per_model);

    /* Resize the set */
    new_size = (*set_size + num_extra_models) * sizeof(oskar_SkyModel);
    *set = realloc((void*)(*set), new_size);
    for (i = *set_size; i < *set_size + num_extra_models; ++i)
    {
        oskar_SkyModel* sky = &((*set)[i]);
        /* Initialise the new models to resize the source fields */
        oskar_sky_model_init(sky, model_type, model_location, max_sources_per_model);
        /* Set the number sources to zero as this is the number currently
         * allocated in the model. */
        sky->num_sources = 0;
    }

    /* Copy sources from the model into the set. */
    /*   Loop over set entries with free space and copy sources into them... */
    number_to_copy = model->num_sources;
    from_offset = 0;
    /*   Declare pointer into the sky model */
    error = oskar_sky_model_init(&model_ptr, model_type, model_location, 0);
    if (error) return error;
    for (i = (*set_size-1 > 0) ? *set_size-1 : 0; i < *set_size + num_extra_models; ++i)
    {
        oskar_SkyModel* sky = &((*set)[i]);
        n = sky->num_sources;
        free_space = max_sources_per_model - n;
        n_copy = MIN(free_space, number_to_copy);
        error = oskar_sky_model_get_ptr(&model_ptr, model, from_offset, n_copy);
        if (error) return error;
        error = oskar_sky_model_insert(sky, &model_ptr, n);
        if (error) return error;
        sky->num_sources = n_copy + n;
        number_to_copy  -= n_copy;
        from_offset     += n_copy;
    }
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    /* Set the use extended flag if needed */
    for (j = (*set_size-1 > 0) ? *set_size-1 : 0; j < *set_size + num_extra_models; ++j)
    {
        oskar_SkyModel* sky = &((*set)[j]);
        for (i = 0; i < sky->num_sources; ++i)
        {
            double FWHM_minor, FWHM_major;
            FWHM_minor = (model_type == OSKAR_DOUBLE) ?
                    ((double*)sky->FWHM_minor.data)[i] :
                    ((float*)sky->FWHM_minor.data)[i];
            FWHM_major = (model_type == OSKAR_DOUBLE) ?
                    ((double*)sky->FWHM_major.data)[i] :
                    ((float*)sky->FWHM_major.data)[i];

            /* If any source in the model is extended set the use extended flag */
            /* __Note__ this assumes that we can't evaluate extended line sources
             * This may not be true in future
             * if oskar_evaluate_gaussian_source_parameters() is updated...
             */
            if (FWHM_minor > 0.0 && FWHM_major > 0.0)
            {
                sky->use_extended = OSKAR_TRUE;
                break;
            }
        }
    }
#endif

    /* Update the set size */
    *set_size += num_extra_models;

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
