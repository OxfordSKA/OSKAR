/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_sky.h>
#include <oskar_mem.h>
#include <oskar_sky_filter_by_flux_cuda.h>

#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_filter_by_flux(oskar_Sky* sky,
        double min_I, double max_I, int* status)
{
    int location;

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
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Get the location. */
    location = oskar_sky_location(sky);
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        oskar_sky_filter_by_flux_cuda(sky, min_I, max_I, status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        int in, out, type, num_sources;
        type = oskar_sky_precision(sky);
        num_sources = oskar_sky_num_sources(sky);

        if (type == OSKAR_SINGLE)
        {
            float *ra_, *dec_, *I_, *Q_, *U_, *V_, *ref_, *spix_, *rm_;
            float *l_, *m_, *n_, *maj_, *min_, *pa_, *a_, *b_, *c_;
            ra_   = oskar_mem_float(oskar_sky_ra(sky), status);
            dec_  = oskar_mem_float(oskar_sky_dec(sky), status);
            I_    = oskar_mem_float(oskar_sky_I(sky), status);
            Q_    = oskar_mem_float(oskar_sky_Q(sky), status);
            U_    = oskar_mem_float(oskar_sky_U(sky), status);
            V_    = oskar_mem_float(oskar_sky_V(sky), status);
            ref_  = oskar_mem_float(oskar_sky_reference_freq(sky), status);
            spix_ = oskar_mem_float(oskar_sky_spectral_index(sky), status);
            rm_   = oskar_mem_float(oskar_sky_rotation_measure(sky), status);
            l_    = oskar_mem_float(oskar_sky_l(sky), status);
            m_    = oskar_mem_float(oskar_sky_m(sky), status);
            n_    = oskar_mem_float(oskar_sky_n(sky), status);
            maj_  = oskar_mem_float(oskar_sky_fwhm_major(sky), status);
            min_  = oskar_mem_float(oskar_sky_fwhm_minor(sky), status);
            pa_   = oskar_mem_float(oskar_sky_position_angle(sky), status);
            a_    = oskar_mem_float(oskar_sky_gaussian_a(sky), status);
            b_    = oskar_mem_float(oskar_sky_gaussian_b(sky), status);
            c_    = oskar_mem_float(oskar_sky_gaussian_c(sky), status);

            for (in = 0, out = 0; in < num_sources; ++in)
            {
                if (!(I_[in] > (float)min_I && I_[in] <= (float)max_I)) continue;
                ra_[out]   = ra_[in];
                dec_[out]  = dec_[in];
                I_[out]    = I_[in];
                Q_[out]    = Q_[in];
                U_[out]    = U_[in];
                V_[out]    = V_[in];
                ref_[out]  = ref_[in];
                spix_[out] = spix_[in];
                rm_[out]   = rm_[in];
                l_[out]    = l_[in];
                m_[out]    = m_[in];
                n_[out]    = n_[in];
                maj_[out]  = maj_[in];
                min_[out]  = min_[in];
                pa_[out]   = pa_[in];
                a_[out]    = a_[in];
                b_[out]    = b_[in];
                c_[out]    = c_[in];
                out++;
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double *ra_, *dec_, *I_, *Q_, *U_, *V_, *ref_, *spix_, *rm_;
            double *l_, *m_, *n_, *maj_, *min_, *pa_, *a_, *b_, *c_;
            ra_   = oskar_mem_double(oskar_sky_ra(sky), status);
            dec_  = oskar_mem_double(oskar_sky_dec(sky), status);
            I_    = oskar_mem_double(oskar_sky_I(sky), status);
            Q_    = oskar_mem_double(oskar_sky_Q(sky), status);
            U_    = oskar_mem_double(oskar_sky_U(sky), status);
            V_    = oskar_mem_double(oskar_sky_V(sky), status);
            ref_  = oskar_mem_double(oskar_sky_reference_freq(sky), status);
            spix_ = oskar_mem_double(oskar_sky_spectral_index(sky), status);
            rm_   = oskar_mem_double(oskar_sky_rotation_measure(sky), status);
            l_    = oskar_mem_double(oskar_sky_l(sky), status);
            m_    = oskar_mem_double(oskar_sky_m(sky), status);
            n_    = oskar_mem_double(oskar_sky_n(sky), status);
            maj_  = oskar_mem_double(oskar_sky_fwhm_major(sky), status);
            min_  = oskar_mem_double(oskar_sky_fwhm_minor(sky), status);
            pa_   = oskar_mem_double(oskar_sky_position_angle(sky), status);
            a_    = oskar_mem_double(oskar_sky_gaussian_a(sky), status);
            b_    = oskar_mem_double(oskar_sky_gaussian_b(sky), status);
            c_    = oskar_mem_double(oskar_sky_gaussian_c(sky), status);

            for (out = 0, in = 0; in < num_sources; ++in)
            {
                if (!(I_[in] > min_I && I_[in] <= max_I)) continue;
                
                ra_[out]   = ra_[in];
                dec_[out]  = dec_[in];
                I_[out]    = I_[in];
                Q_[out]    = Q_[in];
                U_[out]    = U_[in];
                V_[out]    = V_[in];
                ref_[out]  = ref_[in];
                spix_[out] = spix_[in];
                rm_[out]   = rm_[in];
                l_[out]    = l_[in];
                m_[out]    = m_[in];
                n_[out]    = n_[in];
                maj_[out]  = maj_[in];
                min_[out]  = min_[in];
                pa_[out]   = pa_[in];
                a_[out]    = a_[in];
                b_[out]    = b_[in];
                c_[out]    = c_[in];
                out++;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }

        /* Set the new size of the sky model. */
        oskar_sky_resize(sky, out, status);
    }
}

#ifdef __cplusplus
}
#endif
