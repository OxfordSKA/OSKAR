/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_sky_model_horizon_clip.h"
#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_ra_dec_to_hor_lmn_cuda.h"
#include "sky/cudak/oskar_cudak_update_horizon_mask.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_clear_contents.h"
#include "utility/oskar_mem_realloc.h"

#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>

struct is_true {
        __host__ __device__
        bool operator()(const int x) {return (bool)x;}
};

template<typename T>
static void copy_source_data(oskar_SkyModel* output, const oskar_SkyModel* input,
        const oskar_Mem& mask)
{
    int n = input->num_sources;
    thrust::device_ptr<T> out = thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->RA), // Start.
            thrust::device_pointer_cast(((const T*)input->RA) + n), // End.
            thrust::device_pointer_cast((const int*)mask), // Stencil.
            thrust::device_pointer_cast((T*)output->RA), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->Dec), // Start.
            thrust::device_pointer_cast(((const T*)input->Dec) + n), // End.
            thrust::device_pointer_cast((const int*)mask), // Stencil.
            thrust::device_pointer_cast((T*)output->Dec), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->I),
            thrust::device_pointer_cast(((const T*)input->I) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->I), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->Q),
            thrust::device_pointer_cast(((const T*)input->Q) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->Q), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->U),
            thrust::device_pointer_cast(((const T*)input->U) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->U), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->V),
            thrust::device_pointer_cast(((const T*)input->V) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->V), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->reference_freq),
            thrust::device_pointer_cast(((const T*)input->reference_freq) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->reference_freq), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->spectral_index),
            thrust::device_pointer_cast(((const T*)input->spectral_index) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->spectral_index), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->rel_l),
            thrust::device_pointer_cast(((const T*)input->rel_l) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->rel_l), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->rel_m),
            thrust::device_pointer_cast(((const T*)input->rel_m) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->rel_m), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->rel_n),
            thrust::device_pointer_cast(((const T*)input->rel_n) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->rel_n), is_true());

    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->FWHM_major),
            thrust::device_pointer_cast(((const T*)input->FWHM_major) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->FWHM_major), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->FWHM_minor),
            thrust::device_pointer_cast(((const T*)input->FWHM_minor) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->FWHM_minor), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->position_angle),
            thrust::device_pointer_cast(((const T*)input->position_angle) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->position_angle), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->gaussian_a),
            thrust::device_pointer_cast(((const T*)input->gaussian_a) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->gaussian_a), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->gaussian_b),
            thrust::device_pointer_cast(((const T*)input->gaussian_b) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->gaussian_b), is_true());
    thrust::copy_if(
            thrust::device_pointer_cast((const T*)input->gaussian_c),
            thrust::device_pointer_cast(((const T*)input->gaussian_c) + n),
            thrust::device_pointer_cast((const int*)mask),
            thrust::device_pointer_cast((T*)output->gaussian_c), is_true());

    // Get the number of sources above the horizon.
    output->num_sources = out - thrust::device_pointer_cast((T*)output->RA);
}

extern "C"
void oskar_sky_model_horizon_clip(oskar_SkyModel* output,
        const oskar_SkyModel* input, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work, int* status)
{
    /* Check all inputs. */
    if (!output || !input || !telescope || !work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the types match. */
    if (output->type() != input->type())
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check for the correct location. */
    if (output->location() != OSKAR_LOCATION_GPU ||
            input->location() != OSKAR_LOCATION_GPU ||
            work->hor_x.location != OSKAR_LOCATION_GPU)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Copy extended source flag. */
    output->use_extended = input->use_extended;

    /* Get the data dimensions. */
    int num_sources = input->num_sources;
    int num_stations = telescope->num_stations;

    /* Resize the output structure if necessary. */
    if (output->RA.num_elements < num_sources)
        oskar_sky_model_resize(output, num_sources, status);

    /* Resize the work buffers if necessary. */
    if (work->horizon_mask.num_elements < num_sources)
        oskar_mem_realloc(&work->horizon_mask, num_sources, status);
    if (work->hor_x.num_elements < num_sources)
        oskar_mem_realloc(&work->hor_x, num_sources, status);
    if (work->hor_y.num_elements < num_sources)
        oskar_mem_realloc(&work->hor_y, num_sources, status);
    if (work->hor_z.num_elements < num_sources)
        oskar_mem_realloc(&work->hor_z, num_sources, status);

    /* Clear horizon mask. */
    oskar_mem_clear_contents(&work->horizon_mask, status);

    /* Check if safe to proceed. */
    if (*status) return;

    if (input->type() == OSKAR_SINGLE)
    {
        /* Threads per block, blocks. */
        int n_thd = 256;
        int n_blk = (num_sources + n_thd - 1) / n_thd;

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            longitude = telescope->station[i].longitude_rad;
            latitude  = telescope->station[i].latitude_rad;
            lst = gast + longitude;

            /* Evaluate source horizontal l,m,n direction cosines. */
            oskar_ra_dec_to_hor_lmn_cuda_f(num_sources, input->RA,
                    input->Dec, lst, latitude, work->hor_x, work->hor_y, work->hor_z);

            /* Update the mask. */
            oskar_cudak_update_horizon_mask_f OSKAR_CUDAK_CONF(n_blk, n_thd)
            (num_sources, work->hor_z, work->horizon_mask);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<float>(output, input, work->horizon_mask);
        oskar_cuda_check_error(status);
    }
    else if (input->type() == OSKAR_DOUBLE)
    {
        /* Threads per block, blocks. */
        int n_thd = 256;
        int n_blk = (num_sources + n_thd - 1) / n_thd;

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            longitude = telescope->station[i].longitude_rad;
            latitude  = telescope->station[i].latitude_rad;
            lst = gast + longitude;

            /* Evaluate source horizontal l,m,n direction cosines. */
            oskar_ra_dec_to_hor_lmn_cuda_d(num_sources, input->RA,
                    input->Dec, lst, latitude, work->hor_x, work->hor_y, work->hor_z);

            /* Update the mask. */
            oskar_cudak_update_horizon_mask_d OSKAR_CUDAK_CONF(n_blk, n_thd)
            (num_sources, work->hor_z, work->horizon_mask);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<double>(output, input, work->horizon_mask);
        oskar_cuda_check_error(status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}
