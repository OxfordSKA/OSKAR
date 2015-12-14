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

#include <oskar_cross_correlate.h>
#include <oskar_cross_correlate_gaussian_cuda.h>
#include <oskar_cross_correlate_gaussian_omp.h>
#include <oskar_cross_correlate_gaussian_time_smearing_cuda.h>
#include <oskar_cross_correlate_gaussian_time_smearing_omp.h>
#include <oskar_cross_correlate_point_cuda.h>
#include <oskar_cross_correlate_point_omp.h>
#include <oskar_cross_correlate_point_time_smearing_cuda.h>
#include <oskar_cross_correlate_point_time_smearing_omp.h>
#include <oskar_cross_correlate_gaussian_scalar_cuda.h>
#include <oskar_cross_correlate_gaussian_scalar_omp.h>
#include <oskar_cross_correlate_gaussian_time_smearing_scalar_cuda.h>
#include <oskar_cross_correlate_gaussian_time_smearing_scalar_omp.h>
#include <oskar_cross_correlate_point_scalar_cuda.h>
#include <oskar_cross_correlate_point_scalar_omp.h>
#include <oskar_cross_correlate_point_time_smearing_scalar_cuda.h>
#include <oskar_cross_correlate_point_time_smearing_scalar_omp.h>
#include <oskar_cuda_check_error.h>

#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cross_correlate(oskar_Mem* vis, int n_sources, const oskar_Jones* J,
        const oskar_Sky* sky, const oskar_Telescope* tel, const oskar_Mem* u,
        const oskar_Mem* v, const oskar_Mem* w, double gast,
        double frequency_hz, int* status)
{
    int jones_type, base_type, location, matrix_type, n_stations;
    int use_extended;
    double inv_wavelength, frac_bandwidth, time_avg, gha0, dec0;
    double uv_filter_max, uv_filter_min;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data dimensions. */
    n_stations = oskar_telescope_num_stations(tel);
    use_extended = oskar_sky_use_extended(sky);

    /* Get bandwidth-smearing terms. */
    frequency_hz = fabs(frequency_hz);
    inv_wavelength = frequency_hz / 299792458.0;
    frac_bandwidth = oskar_telescope_channel_bandwidth_hz(tel) / frequency_hz;

    /* Get time-average smearing term and Greenwich hour angle. */
    time_avg = oskar_telescope_time_average_sec(tel);
    gha0 = gast - oskar_telescope_phase_centre_ra_rad(tel);
    dec0 = oskar_telescope_phase_centre_dec_rad(tel);

    /* Get UV filter parameters in wavelengths. */
    uv_filter_min = oskar_telescope_uv_filter_min(tel);
    uv_filter_max = oskar_telescope_uv_filter_max(tel);
    if (oskar_telescope_uv_filter_units(tel) == OSKAR_METRES)
    {
        uv_filter_min *= inv_wavelength;
        uv_filter_max *= inv_wavelength;
    }
    if (uv_filter_max < 0.0 || uv_filter_max > FLT_MAX)
        uv_filter_max = FLT_MAX;

    /* Check data locations. */
    location = oskar_sky_mem_location(sky);
    if (oskar_telescope_mem_location(tel) != location ||
            oskar_jones_mem_location(J) != location ||
            oskar_mem_location(vis) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    jones_type = oskar_jones_type(J);
    base_type = oskar_sky_precision(sky);
    matrix_type = oskar_type_is_matrix(jones_type) &&
            oskar_mem_is_matrix(vis);
    if (oskar_mem_precision(vis) != base_type ||
            oskar_type_precision(jones_type) != base_type ||
            oskar_mem_type(u) != base_type || oskar_mem_type(v) != base_type ||
            oskar_mem_type(w) != base_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_type(vis) != jones_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* If neither single or double precision, return error. */
    if (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check the input dimensions. */
    if (oskar_jones_num_sources(J) < n_sources ||
            (int)oskar_mem_length(u) != n_stations ||
            (int)oskar_mem_length(v) != n_stations ||
            (int)oskar_mem_length(w) != n_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check there is enough space for the result. */
    if ((int)oskar_mem_length(vis) < oskar_telescope_num_baselines(tel))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Select kernel. */
    if (base_type == OSKAR_DOUBLE)
    {
        const double *I_, *Q_, *U_, *V_, *l_, *m_, *n_, *a_, *b_, *c_;
        const double *u_, *v_, *w_, *x_, *y_;
        I_ = oskar_mem_double_const(oskar_sky_I_const(sky), status);
        Q_ = oskar_mem_double_const(oskar_sky_Q_const(sky), status);
        U_ = oskar_mem_double_const(oskar_sky_U_const(sky), status);
        V_ = oskar_mem_double_const(oskar_sky_V_const(sky), status);
        l_ = oskar_mem_double_const(oskar_sky_l_const(sky), status);
        m_ = oskar_mem_double_const(oskar_sky_m_const(sky), status);
        n_ = oskar_mem_double_const(oskar_sky_n_const(sky), status);
        a_ = oskar_mem_double_const(oskar_sky_gaussian_a_const(sky), status);
        b_ = oskar_mem_double_const(oskar_sky_gaussian_b_const(sky), status);
        c_ = oskar_mem_double_const(oskar_sky_gaussian_c_const(sky), status);
        u_ = oskar_mem_double_const(u, status);
        v_ = oskar_mem_double_const(v, status);
        w_ = oskar_mem_double_const(w, status);
        x_ = oskar_mem_double_const(
                oskar_telescope_station_true_x_offset_ecef_metres_const(tel),
                status);
        y_ = oskar_mem_double_const(
                oskar_telescope_station_true_y_offset_ecef_metres_const(tel),
                status);

        if (matrix_type)
        {
            double4c *vis_;
            const double4c *J_;
            vis_ = oskar_mem_double4c(vis, status);
            J_   = oskar_jones_double4c_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_cuda_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_cuda_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_cuda_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_cuda_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
                oskar_cuda_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_omp_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_omp_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_omp_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_omp_d
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
            }
        }
        else /* Scalar version. */
        {
            double2 *vis_;
            const double2 *J_;
            vis_ = oskar_mem_double2(vis, status);
            J_   = oskar_jones_double2_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_scalar_cuda_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_scalar_cuda_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_scalar_cuda_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_scalar_cuda_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
                oskar_cuda_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_scalar_omp_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_scalar_omp_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_scalar_omp_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_scalar_omp_d
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
            }
        }
    }
    else /* Single precision. */
    {
        const float *I_, *Q_, *U_, *V_, *l_, *m_, *n_, *a_, *b_, *c_;
        const float *u_, *v_, *w_, *x_, *y_;
        I_ = oskar_mem_float_const(oskar_sky_I_const(sky), status);
        Q_ = oskar_mem_float_const(oskar_sky_Q_const(sky), status);
        U_ = oskar_mem_float_const(oskar_sky_U_const(sky), status);
        V_ = oskar_mem_float_const(oskar_sky_V_const(sky), status);
        l_ = oskar_mem_float_const(oskar_sky_l_const(sky), status);
        m_ = oskar_mem_float_const(oskar_sky_m_const(sky), status);
        n_ = oskar_mem_float_const(oskar_sky_n_const(sky), status);
        a_ = oskar_mem_float_const(oskar_sky_gaussian_a_const(sky), status);
        b_ = oskar_mem_float_const(oskar_sky_gaussian_b_const(sky), status);
        c_ = oskar_mem_float_const(oskar_sky_gaussian_c_const(sky), status);
        u_ = oskar_mem_float_const(u, status);
        v_ = oskar_mem_float_const(v, status);
        w_ = oskar_mem_float_const(w, status);
        x_ = oskar_mem_float_const(
                oskar_telescope_station_true_x_offset_ecef_metres_const(tel),
                status);
        y_ = oskar_mem_float_const(
                oskar_telescope_station_true_y_offset_ecef_metres_const(tel),
                status);

        if (matrix_type)
        {
            float4c *vis_;
            const float4c *J_;
            vis_ = oskar_mem_float4c(vis, status);
            J_   = oskar_jones_float4c_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_cuda_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength,
                                frac_bandwidth, time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_cuda_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_cuda_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_cuda_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
                oskar_cuda_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_omp_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_omp_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_omp_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_omp_f
                        (n_sources, n_stations, J_, I_, Q_, U_, V_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
            }
        }
        else /* Scalar version. */
        {
            float2 *vis_;
            const float2 *J_;
            vis_ = oskar_mem_float2(vis, status);
            J_   = oskar_jones_float2_const(J, status);

            if (location == OSKAR_GPU)
            {
#ifdef OSKAR_HAVE_CUDA
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_scalar_cuda_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_scalar_cuda_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_scalar_cuda_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength,
                                frac_bandwidth, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_scalar_cuda_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
                oskar_cuda_check_error(status);
#else
                *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
            }
            else /* CPU */
            {
                if (time_avg > 0.0)
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_time_smearing_scalar_omp_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength,
                                frac_bandwidth, time_avg, gha0, dec0, vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_time_smearing_scalar_omp_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, x_, y_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                time_avg, gha0, dec0, vis_);
                    }
                }
                else /* Non-time-smearing. */
                {
                    if (use_extended)
                    {
                        oskar_cross_correlate_gaussian_scalar_omp_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                a_, b_, c_, u_, v_, w_, uv_filter_min,
                                uv_filter_max, inv_wavelength, frac_bandwidth,
                                vis_);
                    }
                    else
                    {
                        oskar_cross_correlate_point_scalar_omp_f
                        (n_sources, n_stations, J_, I_, l_, m_, n_,
                                u_, v_, w_, uv_filter_min, uv_filter_max,
                                inv_wavelength, frac_bandwidth, vis_);
                    }
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
