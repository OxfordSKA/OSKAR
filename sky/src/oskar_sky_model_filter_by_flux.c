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
 *    and/or src materials provided with the distribution.
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

#include "oskar_global.h"
#include "sky/oskar_sky_model_filter_by_flux.h"
#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_type.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_filter_by_flux(oskar_SkyModel* sky,
        double min_I, double max_I)
{
    int err = 0, type, location;

    /* Return immediately if no filtering should be done. */
    if (min_I == 0.0 && max_I == 0.0)
        return 0;

    /* Get the type and location. */
    type = oskar_sky_model_type(sky);
    location = oskar_sky_model_location(sky);

    if (location == OSKAR_LOCATION_CPU)
    {
        int in, out;
        if (type == OSKAR_SINGLE)
        {
            float *ra, *dec, *I, *Q, *U, *V, *ref_freq, *spix;
            float *rel_l, *rel_m, *rel_n;
            float *FWHM_major, *FWHM_minor, *position_angle;
            float *gaussian_a, *gaussian_b, *gaussian_c;
            ra       = (float*)sky->RA.data;
            dec      = (float*)sky->Dec.data;
            I        = (float*)sky->I.data;
            Q        = (float*)sky->Q.data;
            U        = (float*)sky->U.data;
            V        = (float*)sky->V.data;
            ref_freq = (float*)sky->reference_freq.data;
            spix     = (float*)sky->spectral_index.data;
            rel_l    = (float*)sky->rel_l.data;
            rel_m    = (float*)sky->rel_m.data;
            rel_n    = (float*)sky->rel_n.data;
            FWHM_major = (float*)sky->FWHM_major.data;
            FWHM_minor = (float*)sky->FWHM_minor.data;
            position_angle = (float*)sky->position_angle.data;
            gaussian_a = (float*)sky->gaussian_a.data;
            gaussian_b = (float*)sky->gaussian_b.data;
            gaussian_c = (float*)sky->gaussian_c.data;

            for (in = 0, out = 0; in < sky->num_sources; ++in)
            {
                if (I[in] < (float)min_I || I[in] > (float)max_I) continue;
                ra[out]       = ra[in];
                dec[out]      = dec[in];
                I[out]        = I[in];
                Q[out]        = Q[in];
                U[out]        = U[in];
                V[out]        = V[in];
                ref_freq[out] = ref_freq[in];
                spix[out]     = spix[in];
                rel_l[out]    = rel_l[in];
                rel_m[out]    = rel_m[in];
                rel_n[out]    = rel_n[in];
                FWHM_major[out] = FWHM_major[in];
                FWHM_minor[out] = FWHM_minor[in];
                position_angle[out] = position_angle[in];
                gaussian_a[out] = gaussian_a[in];
                gaussian_b[out] = gaussian_b[in];
                gaussian_c[out] = gaussian_c[in];

                out++;
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double *ra, *dec, *I, *Q, *U, *V, *ref_freq, *spix;
            double *rel_l, *rel_m, *rel_n;
            double *FWHM_major, *FWHM_minor, *position_angle;
            double *gaussian_a, *gaussian_b, *gaussian_c;
            ra       = (double*)sky->RA.data;
            dec      = (double*)sky->Dec.data;
            I        = (double*)sky->I.data;
            Q        = (double*)sky->Q.data;
            U        = (double*)sky->U.data;
            V        = (double*)sky->V.data;
            ref_freq = (double*)sky->reference_freq.data;
            spix     = (double*)sky->spectral_index.data;
            rel_l    = (double*)sky->rel_l.data;
            rel_m    = (double*)sky->rel_m.data;
            rel_n    = (double*)sky->rel_n.data;
            FWHM_major = (double*)sky->FWHM_major.data;
            FWHM_minor = (double*)sky->FWHM_minor.data;
            position_angle = (double*)sky->position_angle.data;
            gaussian_a = (double*)sky->gaussian_a.data;
            gaussian_b = (double*)sky->gaussian_b.data;
            gaussian_c = (double*)sky->gaussian_c.data;

            for (in = 0, out = 0; in < sky->num_sources; ++in)
            {
                if (I[in] < min_I || I[in] > max_I) continue;
                ra[out]       = ra[in];
                dec[out]      = dec[in];
                I[out]        = I[in];
                Q[out]        = Q[in];
                U[out]        = U[in];
                V[out]        = V[in];
                ref_freq[out] = ref_freq[in];
                spix[out]     = spix[in];
                rel_l[out]    = rel_l[in];
                rel_m[out]    = rel_m[in];
                rel_n[out]    = rel_n[in];
                FWHM_major[out] = FWHM_major[in];
                FWHM_minor[out] = FWHM_minor[in];
                position_angle[out] = position_angle[in];
                gaussian_a[out] = gaussian_a[in];
                gaussian_b[out] = gaussian_b[in];
                gaussian_c[out] = gaussian_c[in];
                out++;
            }
        }
        else
            return OSKAR_ERR_BAD_DATA_TYPE;

        /* Store the new number of sources in the sky model. */
        sky->num_sources = out;
    }
    else
        return OSKAR_ERR_BAD_LOCATION;

    return err;
}

#ifdef __cplusplus
}
#endif
