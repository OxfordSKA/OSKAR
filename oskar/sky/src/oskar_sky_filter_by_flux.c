/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "utility/oskar_device.h"

#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_filter_by_flux(oskar_Sky* sky, double min_I, double max_I,
        int* status)
{
    int in = 0, out = 0;
    if (*status) return;

    /* Return immediately if no filtering should be done. */
    if (min_I <= -DBL_MAX && max_I >= DBL_MAX) return;

    if (max_I < min_I)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Get the meta-data. */
    const int location = oskar_sky_mem_location(sky);
    const int type = oskar_sky_precision(sky);
    const int num_sources = oskar_sky_num_sources(sky);

    /* Filtering is only supported for data in host memory. */
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (type == OSKAR_SINGLE)
    {
        float *ra_ = 0, *dec_ = 0, *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0;
        float *ref_ = 0, *spix_ = 0, *rm_ = 0;
        float *l_ = 0, *m_ = 0, *n_ = 0, *maj_ = 0, *min_ = 0, *pa_ = 0;
        float *a_ = 0, *b_ = 0, *c_ = 0;
        ra_   = oskar_mem_float(oskar_sky_ra_rad(sky), status);
        dec_  = oskar_mem_float(oskar_sky_dec_rad(sky), status);
        I_    = oskar_mem_float(oskar_sky_I(sky), status);
        Q_    = oskar_mem_float(oskar_sky_Q(sky), status);
        U_    = oskar_mem_float(oskar_sky_U(sky), status);
        V_    = oskar_mem_float(oskar_sky_V(sky), status);
        ref_  = oskar_mem_float(oskar_sky_reference_freq_hz(sky), status);
        spix_ = oskar_mem_float(oskar_sky_spectral_index(sky), status);
        rm_   = oskar_mem_float(oskar_sky_rotation_measure_rad(sky), status);
        l_    = oskar_mem_float(oskar_sky_l(sky), status);
        m_    = oskar_mem_float(oskar_sky_m(sky), status);
        n_    = oskar_mem_float(oskar_sky_n(sky), status);
        maj_  = oskar_mem_float(oskar_sky_fwhm_major_rad(sky), status);
        min_  = oskar_mem_float(oskar_sky_fwhm_minor_rad(sky), status);
        pa_   = oskar_mem_float(oskar_sky_position_angle_rad(sky), status);
        a_    = oskar_mem_float(oskar_sky_gaussian_a(sky), status);
        b_    = oskar_mem_float(oskar_sky_gaussian_b(sky), status);
        c_    = oskar_mem_float(oskar_sky_gaussian_c(sky), status);

        for (in = 0; in < num_sources; ++in)
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
        double *ra_ = 0, *dec_ = 0, *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0;
        double *ref_ = 0, *spix_ = 0, *rm_ = 0;
        double *l_ = 0, *m_ = 0, *n_ = 0, *maj_ = 0, *min_ = 0, *pa_ = 0;
        double *a_ = 0, *b_ = 0, *c_ = 0;
        ra_   = oskar_mem_double(oskar_sky_ra_rad(sky), status);
        dec_  = oskar_mem_double(oskar_sky_dec_rad(sky), status);
        I_    = oskar_mem_double(oskar_sky_I(sky), status);
        Q_    = oskar_mem_double(oskar_sky_Q(sky), status);
        U_    = oskar_mem_double(oskar_sky_U(sky), status);
        V_    = oskar_mem_double(oskar_sky_V(sky), status);
        ref_  = oskar_mem_double(oskar_sky_reference_freq_hz(sky), status);
        spix_ = oskar_mem_double(oskar_sky_spectral_index(sky), status);
        rm_   = oskar_mem_double(oskar_sky_rotation_measure_rad(sky), status);
        l_    = oskar_mem_double(oskar_sky_l(sky), status);
        m_    = oskar_mem_double(oskar_sky_m(sky), status);
        n_    = oskar_mem_double(oskar_sky_n(sky), status);
        maj_  = oskar_mem_double(oskar_sky_fwhm_major_rad(sky), status);
        min_  = oskar_mem_double(oskar_sky_fwhm_minor_rad(sky), status);
        pa_   = oskar_mem_double(oskar_sky_position_angle_rad(sky), status);
        a_    = oskar_mem_double(oskar_sky_gaussian_a(sky), status);
        b_    = oskar_mem_double(oskar_sky_gaussian_b(sky), status);
        c_    = oskar_mem_double(oskar_sky_gaussian_c(sky), status);

        for (in = 0; in < num_sources; ++in)
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

#ifdef __cplusplus
}
#endif
