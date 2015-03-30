/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <private_sky.h>
#include <oskar_sky.h>

#include <oskar_sky_copy_source_data.h>
#include <oskar_sky_copy_source_data_cuda.h>
#include <oskar_convert_relative_directions_to_enu_directions.h>
#include <oskar_convert_relative_directions_to_enu_directions_cuda.h>
#include <oskar_update_horizon_mask.h>
#include <oskar_update_horizon_mask_cuda.h>
#include <oskar_cuda_check_error.h>

#define CF(m) oskar_mem_float(m, status)
#define CFC(m) oskar_mem_float_const(m, status)
#define CD(m) oskar_mem_double(m, status)
#define CDC(m) oskar_mem_double_const(m, status)

#ifdef __cplusplus
extern "C" {
#endif

static void horizon_clip_single(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, int location, double ra0,
        double dec0, double gast, oskar_Mem* hor_x, oskar_Mem* hor_y,
        oskar_Mem* hor_z, int* mask, int* num_out, int* status);
static void horizon_clip_double(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, int location, double ra0,
        double dec0, double gast, oskar_Mem* hor_x, oskar_Mem* hor_y,
        oskar_Mem* hor_z, int* mask, int* num_out, int* status);
static double ha0(double longitude, double ra0, double gast);

void oskar_sky_horizon_clip(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, double gast,
        oskar_StationWork* work, int* status)
{
    int *mask, type, location, num_in, num_out = 0;
    oskar_Mem *horizon_mask, *hor_x, *hor_y, *hor_z;
    double ra0, dec0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pointers to work arrays. */
    horizon_mask = oskar_station_work_horizon_mask(work);
    hor_x = oskar_station_work_enu_direction_x(work);
    hor_y = oskar_station_work_enu_direction_y(work);
    hor_z = oskar_station_work_enu_direction_z(work);

    /* Check that the types match. */
    type = oskar_sky_precision(out);
    if (oskar_sky_precision(in) != type ||
            oskar_mem_type(hor_x) != type ||
            oskar_mem_type(hor_y) != type ||
            oskar_mem_type(hor_z) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check that the locations match. */
    location = oskar_sky_mem_location(out);
    if (oskar_sky_mem_location(in) != location ||
            oskar_mem_location(horizon_mask) != location ||
            oskar_mem_location(hor_x) != location ||
            oskar_mem_location(hor_y) != location ||
            oskar_mem_location(hor_z) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Get remaining properties of input sky model. */
    num_in = oskar_sky_num_sources(in);
    ra0 = oskar_sky_reference_ra_rad(in);
    dec0 = oskar_sky_reference_dec_rad(in);

    /* Copy meta-data. */
    oskar_sky_set_use_extended(out, oskar_sky_use_extended(in));
    out->reference_ra_rad = ra0;
    out->reference_dec_rad = dec0;

    /* Resize the output structure if necessary. */
    if (oskar_sky_capacity(out) < num_in)
        oskar_sky_resize(out, num_in, status);

    /* Resize the work buffers if necessary. */
    if ((int)oskar_mem_length(horizon_mask) < num_in)
        oskar_mem_realloc(horizon_mask, num_in, status);
    if ((int)oskar_mem_length(hor_x) < num_in)
        oskar_mem_realloc(hor_x, num_in, status);
    if ((int)oskar_mem_length(hor_y) < num_in)
        oskar_mem_realloc(hor_y, num_in, status);
    if ((int)oskar_mem_length(hor_z) < num_in)
        oskar_mem_realloc(hor_z, num_in, status);

    /* Clear horizon mask. */
    oskar_mem_clear_contents(horizon_mask, status);
    mask = oskar_mem_int(horizon_mask, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Single or double precision? */
    if (type == OSKAR_SINGLE)
    {
        horizon_clip_single(out, in, telescope, location, ra0, dec0, gast,
                hor_x, hor_y, hor_z, mask, &num_out, status);
    }
    else if (type == OSKAR_DOUBLE)
    {
        horizon_clip_double(out, in, telescope, location, ra0, dec0, gast,
                hor_x, hor_y, hor_z, mask, &num_out, status);
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* Set the number of sources in the output sky model. */
    out->num_sources = num_out;
}


static void horizon_clip_single(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, int location, double ra0,
        double dec0, double gast, oskar_Mem* hor_x, oskar_Mem* hor_y,
        oskar_Mem* hor_z, int* mask, int* num_out, int* status)
{
    const float *ra, *dec, *I, *Q, *U, *V;
    const float *ref, *sp, *rm, *l, *m, *n;
    const float *a, *b, *c, *maj, *min, *pa;
    float *o_ra, *o_dec, *o_I, *o_Q, *o_U, *o_V;
    float *o_ref, *o_sp, *o_rm, *o_l, *o_m, *o_n;
    float *o_a, *o_b, *o_c, *o_maj, *o_min, *o_pa;
    float *x, *y, *z;
    int i, num_in, num_stations;
    const oskar_Station* s;

    /* Dimensions. */
    num_in = oskar_sky_num_sources(in);
    num_stations = oskar_telescope_num_stations(telescope);

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

    /* Work arrays. */
    x = CF(hor_x);
    y = CF(hor_y);
    z = CF(hor_z);

    /* Check data location. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Create the mask. */
        for (i = 0; i < num_stations; ++i)
        {
            /* Evaluate source horizontal ENU direction cosines. */
            s = oskar_telescope_station_const(telescope, i);
            oskar_convert_relative_directions_to_enu_directions_cuda_f(
                    x, y, z, num_in, l, m, n,
                    ha0(oskar_station_lon_rad(s), ra0, gast), dec0,
                    oskar_station_lat_rad(s));

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_f(num_in, mask, z);
        }

        /* Copy out source data based on the mask values. */
        oskar_sky_copy_source_data_cuda_f(num_in, num_out, mask,
                ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        /* Create the mask. */
        for (i = 0; i < num_stations; ++i)
        {
            /* Evaluate source horizontal ENU direction cosines. */
            s = oskar_telescope_station_const(telescope, i);
            oskar_convert_relative_directions_to_enu_directions_f(
                    x, y, z, num_in, l, m, n,
                    ha0(oskar_station_lon_rad(s), ra0, gast), dec0,
                    oskar_station_lat_rad(s));

            /* Update the mask. */
            oskar_update_horizon_mask_f(num_in, mask, z);
        }

        /* Copy out source data based on the mask values. */
        oskar_sky_copy_source_data_f(num_in, num_out, mask,
                ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

static void horizon_clip_double(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, int location, double ra0,
        double dec0, double gast, oskar_Mem* hor_x, oskar_Mem* hor_y,
        oskar_Mem* hor_z, int* mask, int* num_out, int* status)
{
    const double *ra, *dec, *I, *Q, *U, *V;
    const double *ref, *sp, *rm, *l, *m, *n;
    const double *a, *b, *c, *maj, *min, *pa;
    double *o_ra, *o_dec, *o_I, *o_Q, *o_U, *o_V;
    double *o_ref, *o_sp, *o_rm, *o_l, *o_m, *o_n;
    double *o_a, *o_b, *o_c, *o_maj, *o_min, *o_pa;
    double *x, *y, *z;
    int i, num_in, num_stations;
    const oskar_Station* s;

    /* Dimensions. */
    num_in = oskar_sky_num_sources(in);
    num_stations = oskar_telescope_num_stations(telescope);

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

    /* Work arrays. */
    x = CD(hor_x);
    y = CD(hor_y);
    z = CD(hor_z);

    /* Check data location. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Create the mask. */
        for (i = 0; i < num_stations; ++i)
        {
            /* Evaluate source horizontal ENU direction cosines. */
            s = oskar_telescope_station_const(telescope, i);
            oskar_convert_relative_directions_to_enu_directions_cuda_d(
                    x, y, z, num_in, l, m, n,
                    ha0(oskar_station_lon_rad(s), ra0, gast), dec0,
                    oskar_station_lat_rad(s));

            /* Update the mask. */
            oskar_update_horizon_mask_cuda_d(num_in, mask, z);
        }

        /* Copy out source data based on the mask values. */
        oskar_sky_copy_source_data_cuda_d(num_in, num_out, mask,
                ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        /* Create the mask. */
        for (i = 0; i < num_stations; ++i)
        {
            /* Evaluate source horizontal ENU direction cosines. */
            s = oskar_telescope_station_const(telescope, i);
            oskar_convert_relative_directions_to_enu_directions_d(
                    x, y, z, num_in, l, m, n,
                    ha0(oskar_station_lon_rad(s), ra0, gast), dec0,
                    oskar_station_lat_rad(s));

            /* Update the mask. */
            oskar_update_horizon_mask_d(num_in, mask, z);
        }

        /* Copy out source data based on the mask values. */
        oskar_sky_copy_source_data_d(num_in, num_out, mask,
                ra, o_ra, dec, o_dec, I, o_I, Q, o_Q, U, o_U, V, o_V,
                ref, o_ref, sp, o_sp, rm, o_rm, l, o_l, m, o_m, n, o_n,
                a, o_a, b, o_b, c, o_c, maj, o_maj, min, o_min, pa, o_pa);
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

static double ha0(double longitude, double ra0, double gast)
{
    return (gast + longitude) - ra0;
}

#ifdef __cplusplus
}
#endif
