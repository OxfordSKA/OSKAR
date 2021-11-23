/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "utility/oskar_device.h"

#define CB(m) oskar_mem_buffer(m)
#define CBC(m) oskar_mem_buffer_const(m)
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
    int i = 0, num_in = 0, num_out = 0;
    if (*status) return;
    const int location = oskar_sky_mem_location(in);
    const int type = oskar_sky_precision(out);
    num_in = oskar_sky_num_sources(in);
    if (location == OSKAR_CPU)
    {
        const int* mask = oskar_mem_int_const(horizon_mask, status);
        (void) indices;
        switch (type)
        {
        case OSKAR_SINGLE:
        {
            const float *ra = 0, *dec = 0, *I = 0, *Q = 0, *U = 0, *V = 0;
            const float *ref = 0, *sp = 0, *rm = 0, *l = 0, *m = 0, *n = 0;
            const float *a = 0, *b = 0, *c = 0, *maj = 0, *min = 0, *pa = 0;
            float *o_ra = 0, *o_dec = 0, *o_I = 0, *o_Q = 0, *o_U = 0, *o_V = 0;
            float *o_ref = 0, *o_sp = 0, *o_rm = 0, *o_l = 0, *o_m = 0, *o_n = 0;
            float *o_a = 0, *o_b = 0, *o_c = 0, *o_maj = 0, *o_min = 0, *o_pa = 0;

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
            COPY_SOURCE_DATA
            break;
        }
        case OSKAR_DOUBLE:
        {
            const double *ra = 0, *dec = 0, *I = 0, *Q = 0, *U = 0, *V = 0;
            const double *ref = 0, *sp = 0, *rm = 0, *l = 0, *m = 0, *n = 0;
            const double *a = 0, *b = 0, *c = 0, *maj = 0, *min = 0, *pa = 0;
            double *o_ra = 0, *o_dec = 0, *o_I = 0, *o_Q = 0, *o_U = 0, *o_V = 0;
            double *o_ref = 0, *o_sp = 0, *o_rm = 0, *o_l = 0, *o_m = 0, *o_n = 0;
            double *o_a = 0, *o_b = 0, *o_c = 0, *o_maj = 0, *o_min = 0, *o_pa = 0;

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
            COPY_SOURCE_DATA
            break;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_DOUBLE)
        {
            k = "copy_source_data_double";
        }
        else if (type == OSKAR_SINGLE)
        {
            k = "copy_source_data_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_in, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_in},
                {PTR_SZ, oskar_mem_buffer_const(horizon_mask)},
                {PTR_SZ, oskar_mem_buffer_const(indices)},
                {PTR_SZ, CBC(oskar_sky_ra_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_ra_rad(out))},
                {PTR_SZ, CBC(oskar_sky_dec_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_dec_rad(out))},
                {PTR_SZ, CBC(oskar_sky_I_const(in))},
                {PTR_SZ, CB(oskar_sky_I(out))},
                {PTR_SZ, CBC(oskar_sky_Q_const(in))},
                {PTR_SZ, CB(oskar_sky_Q(out))},
                {PTR_SZ, CBC(oskar_sky_U_const(in))},
                {PTR_SZ, CB(oskar_sky_U(out))},
                {PTR_SZ, CBC(oskar_sky_V_const(in))},
                {PTR_SZ, CB(oskar_sky_V(out))},
                {PTR_SZ, CBC(oskar_sky_reference_freq_hz_const(in))},
                {PTR_SZ, CB(oskar_sky_reference_freq_hz(out))},
                {PTR_SZ, CBC(oskar_sky_spectral_index_const(in))},
                {PTR_SZ, CB(oskar_sky_spectral_index(out))},
                {PTR_SZ, CBC(oskar_sky_rotation_measure_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_rotation_measure_rad(out))},
                {PTR_SZ, CBC(oskar_sky_l_const(in))},
                {PTR_SZ, CB(oskar_sky_l(out))},
                {PTR_SZ, CBC(oskar_sky_m_const(in))},
                {PTR_SZ, CB(oskar_sky_m(out))},
                {PTR_SZ, CBC(oskar_sky_n_const(in))},
                {PTR_SZ, CB(oskar_sky_n(out))},
                {PTR_SZ, CBC(oskar_sky_gaussian_a_const(in))},
                {PTR_SZ, CB(oskar_sky_gaussian_a(out))},
                {PTR_SZ, CBC(oskar_sky_gaussian_b_const(in))},
                {PTR_SZ, CB(oskar_sky_gaussian_b(out))},
                {PTR_SZ, CBC(oskar_sky_gaussian_c_const(in))},
                {PTR_SZ, CB(oskar_sky_gaussian_c(out))},
                {PTR_SZ, CBC(oskar_sky_fwhm_major_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_fwhm_major_rad(out))},
                {PTR_SZ, CBC(oskar_sky_fwhm_minor_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_fwhm_minor_rad(out))},
                {PTR_SZ, CBC(oskar_sky_position_angle_rad_const(in))},
                {PTR_SZ, CB(oskar_sky_position_angle_rad(out))}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        /* Last element of index array is the total number copied. */
        oskar_mem_read_element(indices, num_in, &num_out, status);
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
