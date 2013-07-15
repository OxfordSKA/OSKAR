/*
 * Copyright (c) 2013, The University of Oxford
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

#include "sky/oskar_sky_model_filter_by_flux_cuda.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_type.h"

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <float.h>

using thrust::remove_if;
using thrust::device_pointer_cast;

template<typename T>
struct is_outside_range {
    __host__ __device__
    bool operator()(const T x) { return (x > max_f || x < min_f); }
    T min_f;
    T max_f;
};

template<typename T>
static void filter_source_data(oskar_SkyModel* output, T min_f, T max_f)
{
    int n = output->num_sources;
    is_outside_range<T> range_check;
    range_check.min_f = min_f;
    range_check.max_f = max_f;
    thrust::device_ptr<T> out = remove_if(
            device_pointer_cast((T*)output->RA.data), // Start.
            device_pointer_cast(((T*)output->RA.data) + n), // End.
            device_pointer_cast((const T*)output->I.data), // Stencil.
            range_check);
    remove_if(
            device_pointer_cast((T*)output->Dec.data), // Start.
            device_pointer_cast(((T*)output->Dec.data) + n), // End.
            device_pointer_cast((const T*)output->I.data), // Stencil.
            range_check);
    remove_if(
            device_pointer_cast((T*)output->Q.data),
            device_pointer_cast(((T*)output->Q.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->U.data),
            device_pointer_cast(((T*)output->U.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->V.data),
            device_pointer_cast(((T*)output->V.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->reference_freq.data),
            device_pointer_cast(((T*)output->reference_freq.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->spectral_index.data),
            device_pointer_cast(((T*)output->spectral_index.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->l.data),
            device_pointer_cast(((T*)output->l.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->m.data),
            device_pointer_cast(((T*)output->m.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->n.data),
            device_pointer_cast(((T*)output->n.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);

    remove_if(
            device_pointer_cast((T*)output->FWHM_major.data),
            device_pointer_cast(((T*)output->FWHM_major.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->FWHM_minor.data),
            device_pointer_cast(((T*)output->FWHM_minor.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->position_angle.data),
            device_pointer_cast(((T*)output->position_angle.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->gaussian_a.data),
            device_pointer_cast(((T*)output->gaussian_a.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->gaussian_b.data),
            device_pointer_cast(((T*)output->gaussian_b.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);
    remove_if(
            device_pointer_cast((T*)output->gaussian_c.data),
            device_pointer_cast(((T*)output->gaussian_c.data) + n),
            device_pointer_cast((const T*)output->I.data),
            range_check);

    // Finally, remove Stokes I values.
    remove_if(
            device_pointer_cast((T*)output->I.data),
            device_pointer_cast(((T*)output->I.data) + n),
            device_pointer_cast((T*)output->I.data),
            range_check);

    // Get the number of sources in the new sky model.
    output->num_sources = out - device_pointer_cast((T*)output->RA);
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_model_filter_by_flux_cuda(oskar_SkyModel* sky,
        double min_I, double max_I, int* status)
{
    int type, location;

    /* Return immediately if no filtering should be done. */
    if (min_I <= 0.0 && max_I <= 0.0)
        return;

    /* If only the lower limit is set */
    if (max_I <= 0.0 && min_I > 0.0)
        max_I = DBL_MAX;

    /* If only the upper limit is set */
    if (min_I <= 0.0 && max_I > 0.0)
        min_I = 0.0;

    if (max_I < min_I)
    {
        *status = OSKAR_ERR_SETUP_FAIL;
        return;
    }

    /* Get the type and location. */
    type = oskar_sky_model_type(sky);
    location = oskar_sky_model_location(sky);

    /* Check location. */
    if (location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (type == OSKAR_SINGLE)
    {
        filter_source_data<float>(sky, (float)min_I, (float)max_I);
    }
    else if (type == OSKAR_DOUBLE)
    {
        filter_source_data<double>(sky, min_I, max_I);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
