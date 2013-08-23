/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_sky_model_horizon_clip.h>
#include <oskar_sky_model_resize.h>
#include <oskar_ra_dec_to_hor_lmn_cuda.h>
#include <oskar_update_horizon_mask_cuda.h>
#include <oskar_cuda_check_error.h>
#include <oskar_mem_clear_contents.h>
#include <oskar_mem_realloc.h>
#include <oskar_mem_to_type.h>

#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>

using thrust::copy_if;
using thrust::device_pointer_cast;

struct is_true {
        __host__ __device__
        bool operator()(const int x) {return (bool)x;}
};

template<typename T>
static void copy_source_data(oskar_SkyModel* output,
        const oskar_SkyModel* input, const oskar_Mem* mask, int* status)
{
    int n = input->num_sources;
    const T* ra_in   = (const T*)oskar_mem_to_const_void(&input->RA);
    T* ra_out        = (T*)oskar_mem_to_void(&output->RA);
    const T* dec_in  = (const T*)oskar_mem_to_const_void(&input->Dec);
    T* dec_out       = (T*)oskar_mem_to_void(&output->Dec);
    const T* I_in    = (const T*)oskar_mem_to_const_void(&input->I);
    T* I_out         = (T*)oskar_mem_to_void(&output->I);
    const T* Q_in    = (const T*)oskar_mem_to_const_void(&input->Q);
    T* Q_out         = (T*)oskar_mem_to_void(&output->Q);
    const T* U_in    = (const T*)oskar_mem_to_const_void(&input->U);
    T* U_out         = (T*)oskar_mem_to_void(&output->U);
    const T* V_in    = (const T*)oskar_mem_to_const_void(&input->V);
    T* V_out         = (T*)oskar_mem_to_void(&output->V);
    const T* ref_in  = (const T*)oskar_mem_to_const_void(&input->reference_freq);
    T* ref_out       = (T*)oskar_mem_to_void(&output->reference_freq);
    const T* sp_in   = (const T*)oskar_mem_to_const_void(&input->spectral_index);
    T* sp_out        = (T*)oskar_mem_to_void(&output->spectral_index);
    const T* l_in    = (const T*)oskar_mem_to_const_void(&input->l);
    T* l_out         = (T*)oskar_mem_to_void(&output->l);
    const T* m_in    = (const T*)oskar_mem_to_const_void(&input->m);
    T* m_out         = (T*)oskar_mem_to_void(&output->m);
    const T* n_in    = (const T*)oskar_mem_to_const_void(&input->n);
    T* n_out         = (T*)oskar_mem_to_void(&output->n);
    const T* a_in    = (const T*)oskar_mem_to_const_void(&input->gaussian_a);
    T* a_out         = (T*)oskar_mem_to_void(&output->gaussian_a);
    const T* b_in    = (const T*)oskar_mem_to_const_void(&input->gaussian_b);
    T* b_out         = (T*)oskar_mem_to_void(&output->gaussian_b);
    const T* c_in    = (const T*)oskar_mem_to_const_void(&input->gaussian_c);
    T* c_out         = (T*)oskar_mem_to_void(&output->gaussian_c);
    const T* maj_in  = (const T*)oskar_mem_to_const_void(&input->FWHM_major);
    T* maj_out       = (T*)oskar_mem_to_void(&output->FWHM_major);
    const T* min_in  = (const T*)oskar_mem_to_const_void(&input->FWHM_minor);
    T* min_out       = (T*)oskar_mem_to_void(&output->FWHM_minor);
    const T* pa_in   = (const T*)oskar_mem_to_const_void(&input->position_angle);
    T* pa_out        = (T*)oskar_mem_to_void(&output->position_angle);
    thrust::device_ptr<const int> m = device_pointer_cast(
            oskar_mem_to_const_int(mask, status));
    thrust::device_ptr<T> out = copy_if(device_pointer_cast(ra_in),
            device_pointer_cast(ra_in + n), m,
            device_pointer_cast(ra_out), is_true());
    copy_if(device_pointer_cast(dec_in), device_pointer_cast(dec_in + n), m,
            device_pointer_cast(dec_out), is_true());
    copy_if(device_pointer_cast(I_in), device_pointer_cast(I_in + n), m,
            device_pointer_cast(I_out), is_true());
    copy_if(device_pointer_cast(Q_in), device_pointer_cast(Q_in + n), m,
            device_pointer_cast(Q_out), is_true());
    copy_if(device_pointer_cast(U_in), device_pointer_cast(U_in + n), m,
            device_pointer_cast(U_out), is_true());
    copy_if(device_pointer_cast(V_in), device_pointer_cast(V_in + n), m,
            device_pointer_cast(V_out), is_true());
    copy_if(device_pointer_cast(ref_in), device_pointer_cast(ref_in + n), m,
            device_pointer_cast(ref_out), is_true());
    copy_if(device_pointer_cast(sp_in), device_pointer_cast(sp_in + n), m,
            device_pointer_cast(sp_out), is_true());
    copy_if(device_pointer_cast(l_in), device_pointer_cast(l_in + n), m,
            device_pointer_cast(l_out), is_true());
    copy_if(device_pointer_cast(m_in), device_pointer_cast(m_in + n), m,
            device_pointer_cast(m_out), is_true());
    copy_if(device_pointer_cast(n_in), device_pointer_cast(n_in + n), m,
            device_pointer_cast(n_out), is_true());

    copy_if(device_pointer_cast(maj_in), device_pointer_cast(maj_in + n), m,
            device_pointer_cast(maj_out), is_true());
    copy_if(device_pointer_cast(min_in), device_pointer_cast(min_in + n), m,
            device_pointer_cast(min_out), is_true());
    copy_if(device_pointer_cast(pa_in), device_pointer_cast(pa_in + n), m,
            device_pointer_cast(pa_out), is_true());
    copy_if(device_pointer_cast(a_in), device_pointer_cast(a_in + n), m,
            device_pointer_cast(a_out), is_true());
    copy_if(device_pointer_cast(b_in), device_pointer_cast(b_in + n), m,
            device_pointer_cast(b_out), is_true());
    copy_if(device_pointer_cast(c_in), device_pointer_cast(c_in + n), m,
            device_pointer_cast(c_out), is_true());

    // Get the number of sources above the horizon.
    output->num_sources = out - device_pointer_cast(ra_out);
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
        const float *ra, *dec;
        float *x, *y, *z;
        int *mask;
        ra   = oskar_mem_to_const_float(&input->RA, status);
        dec  = oskar_mem_to_const_float(&input->Dec, status);
        x    = oskar_mem_to_float(&work->hor_x, status);
        y    = oskar_mem_to_float(&work->hor_y, status);
        z    = oskar_mem_to_float(&work->hor_z, status);
        mask = oskar_mem_to_int(&work->horizon_mask, status);

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            longitude = telescope->station[i].longitude_rad;
            latitude  = telescope->station[i].latitude_rad;
            lst = gast + longitude;

            /* Evaluate source horizontal x,y,z direction cosines. */
            oskar_ra_dec_to_hor_lmn_cuda_f(num_sources, ra, dec, lst, latitude,
                    x, y, z);

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_f(num_sources, mask,
                    (const float*)z);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<float>(output, input, &work->horizon_mask, status);
        oskar_cuda_check_error(status);
    }
    else if (input->type() == OSKAR_DOUBLE)
    {
        const double *ra, *dec;
        double *x, *y, *z;
        int *mask;
        ra   = oskar_mem_to_const_double(&input->RA, status);
        dec  = oskar_mem_to_const_double(&input->Dec, status);
        x    = oskar_mem_to_double(&work->hor_x, status);
        y    = oskar_mem_to_double(&work->hor_y, status);
        z    = oskar_mem_to_double(&work->hor_z, status);
        mask = oskar_mem_to_int(&work->horizon_mask, status);

        /* Create the mask. */
        for (int i = 0; i < num_stations; ++i)
        {
            /* Get the station position. */
            double longitude, latitude, lst;
            longitude = telescope->station[i].longitude_rad;
            latitude  = telescope->station[i].latitude_rad;
            lst = gast + longitude;

            /* Evaluate source horizontal x,y,z direction cosines. */
            oskar_ra_dec_to_hor_lmn_cuda_d(num_sources, ra, dec, lst, latitude,
                    x, y, z);

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_d(num_sources, mask,
                    (const double*)z);
        }

        /* Copy out source data based on the mask values. */
        copy_source_data<double>(output, input, &work->horizon_mask, status);
        oskar_cuda_check_error(status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}
