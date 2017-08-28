/*
 * Copyright (c) 2014-2017, The University of Oxford
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

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "sky/oskar_sky_copy_source_data_cuda.h"
#include "utility/oskar_device_utils.h"

#define CF(m) oskar_mem_float(m, status)
#define CFC(m) oskar_mem_float_const(m, status)
#define CD(m) oskar_mem_double(m, status)
#define CDC(m) oskar_mem_double_const(m, status)

#ifdef __cplusplus
extern "C" {
#endif

#define COPY_SOURCE_DATA \
        for (i = 0; i < num_in; ++i) \
            if (mask[i]) \
            { \
                o_ra[num_out]  = ra[i]; \
                o_dec[num_out] = dec[i]; \
                o_I[num_out]   = I[i]; \
                o_Q[num_out]   = Q[i]; \
                o_U[num_out]   = U[i]; \
                o_V[num_out]   = V[i]; \
                o_ref[num_out] = ref[i]; \
                o_sp[num_out]  = sp[i]; \
                o_rm[num_out]  = rm[i]; \
                o_l[num_out]   = l[i]; \
                o_m[num_out]   = m[i]; \
                o_n[num_out]   = n[i]; \
                o_a[num_out]   = a[i]; \
                o_b[num_out]   = b[i]; \
                o_c[num_out]   = c[i]; \
                o_maj[num_out] = maj[i]; \
                o_min[num_out] = min[i]; \
                o_pa[num_out]  = pa[i]; \
                num_out++; \
            }

void oskar_sky_copy_source_data(const oskar_Sky* in,
        const oskar_Mem* horizon_mask, const oskar_Mem* indices,
        oskar_Sky* out, int* status)
{
    int i, num_in, num_out = 0, type;
    const int *mask;
    if (*status) return;
    type = oskar_sky_precision(out);
    mask = oskar_mem_int_const(horizon_mask, status);
    num_in = oskar_sky_num_sources(in);
    switch (type)
    {
    case OSKAR_SINGLE:
    {
        const float *ra, *dec, *I, *Q, *U, *V;
        const float *ref, *sp, *rm, *l, *m, *n;
        const float *a, *b, *c, *maj, *min, *pa;
        float *o_ra, *o_dec, *o_I, *o_Q, *o_U, *o_V;
        float *o_ref, *o_sp, *o_rm, *o_l, *o_m, *o_n;
        float *o_a, *o_b, *o_c, *o_maj, *o_min, *o_pa;

        /* Inputs. */
        ra = CFC(oskar_sky_ra_rad_const(in));
        dec = CFC(oskar_sky_dec_rad_const(in));
        I = CFC(oskar_sky_I_const(in));
        Q = CFC(oskar_sky_Q_const(in));
        U = CFC(oskar_sky_U_const(in));
        V = CFC(oskar_sky_V_const(in));
        ref = CFC(oskar_sky_reference_freq_hz_const(in));
        sp = CFC(oskar_sky_spectral_index_const(in));
        rm = CFC(oskar_sky_rotation_measure_rad_const(in));
        l = CFC(oskar_sky_l_const(in));
        m = CFC(oskar_sky_m_const(in));
        n = CFC(oskar_sky_n_const(in));
        a = CFC(oskar_sky_gaussian_a_const(in));
        b = CFC(oskar_sky_gaussian_b_const(in));
        c = CFC(oskar_sky_gaussian_c_const(in));
        maj = CFC(oskar_sky_fwhm_major_rad_const(in));
        min = CFC(oskar_sky_fwhm_minor_rad_const(in));
        pa = CFC(oskar_sky_position_angle_rad_const(in));

        /* Outputs. */
        o_ra = CF(oskar_sky_ra_rad(out));
        o_dec = CF(oskar_sky_dec_rad(out));
        o_I = CF(oskar_sky_I(out));
        o_Q = CF(oskar_sky_Q(out));
        o_U = CF(oskar_sky_U(out));
        o_V = CF(oskar_sky_V(out));
        o_ref = CF(oskar_sky_reference_freq_hz(out));
        o_sp = CF(oskar_sky_spectral_index(out));
        o_rm = CF(oskar_sky_rotation_measure_rad(out));
        o_l = CF(oskar_sky_l(out));
        o_m = CF(oskar_sky_m(out));
        o_n = CF(oskar_sky_n(out));
        o_a = CF(oskar_sky_gaussian_a(out));
        o_b = CF(oskar_sky_gaussian_b(out));
        o_c = CF(oskar_sky_gaussian_c(out));
        o_maj = CF(oskar_sky_fwhm_major_rad(out));
        o_min = CF(oskar_sky_fwhm_minor_rad(out));
        o_pa = CF(oskar_sky_position_angle_rad(out));

        /* Copy source data. */
        if (oskar_sky_mem_location(in) == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_sky_copy_source_data_cuda_f(num_in, &num_out, mask,
                    oskar_mem_int_const(indices, status),
                    ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                    ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                    a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (oskar_sky_mem_location(in) == OSKAR_CPU)
        {
            (void) indices;
            COPY_SOURCE_DATA
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
        break;
    }
    case OSKAR_DOUBLE:
    {
        const double *ra, *dec, *I, *Q, *U, *V;
        const double *ref, *sp, *rm, *l, *m, *n;
        const double *a, *b, *c, *maj, *min, *pa;
        double *o_ra, *o_dec, *o_I, *o_Q, *o_U, *o_V;
        double *o_ref, *o_sp, *o_rm, *o_l, *o_m, *o_n;
        double *o_a, *o_b, *o_c, *o_maj, *o_min, *o_pa;

        /* Inputs. */
        ra = CDC(oskar_sky_ra_rad_const(in));
        dec = CDC(oskar_sky_dec_rad_const(in));
        I = CDC(oskar_sky_I_const(in));
        Q = CDC(oskar_sky_Q_const(in));
        U = CDC(oskar_sky_U_const(in));
        V = CDC(oskar_sky_V_const(in));
        ref = CDC(oskar_sky_reference_freq_hz_const(in));
        sp = CDC(oskar_sky_spectral_index_const(in));
        rm = CDC(oskar_sky_rotation_measure_rad_const(in));
        l = CDC(oskar_sky_l_const(in));
        m = CDC(oskar_sky_m_const(in));
        n = CDC(oskar_sky_n_const(in));
        a = CDC(oskar_sky_gaussian_a_const(in));
        b = CDC(oskar_sky_gaussian_b_const(in));
        c = CDC(oskar_sky_gaussian_c_const(in));
        maj = CDC(oskar_sky_fwhm_major_rad_const(in));
        min = CDC(oskar_sky_fwhm_minor_rad_const(in));
        pa = CDC(oskar_sky_position_angle_rad_const(in));

        /* Outputs. */
        o_ra = CD(oskar_sky_ra_rad(out));
        o_dec = CD(oskar_sky_dec_rad(out));
        o_I = CD(oskar_sky_I(out));
        o_Q = CD(oskar_sky_Q(out));
        o_U = CD(oskar_sky_U(out));
        o_V = CD(oskar_sky_V(out));
        o_ref = CD(oskar_sky_reference_freq_hz(out));
        o_sp = CD(oskar_sky_spectral_index(out));
        o_rm = CD(oskar_sky_rotation_measure_rad(out));
        o_l = CD(oskar_sky_l(out));
        o_m = CD(oskar_sky_m(out));
        o_n = CD(oskar_sky_n(out));
        o_a = CD(oskar_sky_gaussian_a(out));
        o_b = CD(oskar_sky_gaussian_b(out));
        o_c = CD(oskar_sky_gaussian_c(out));
        o_maj = CD(oskar_sky_fwhm_major_rad(out));
        o_min = CD(oskar_sky_fwhm_minor_rad(out));
        o_pa = CD(oskar_sky_position_angle_rad(out));

        /* Copy source data. */
        if (oskar_sky_mem_location(in) == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_sky_copy_source_data_cuda_d(num_in, &num_out, mask,
                    oskar_mem_int_const(indices, status),
                    ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                    ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                    a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (oskar_sky_mem_location(in) == OSKAR_CPU)
        {
            (void) indices;
            COPY_SOURCE_DATA
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
        break;
    }
    default:
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        break;
    }

    /* Copy metadata. */
    out->use_extended = in->use_extended;
    out->reference_ra_rad = in->reference_ra_rad;
    out->reference_dec_rad = in->reference_dec_rad;

    /* Set the number of sources in the output sky model. */
    out->num_sources = num_out;
}


#ifdef __cplusplus
}
#endif
