/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "correlate/oskar_cross_correlate.h"
#include "correlate/oskar_cross_correlate_cuda.h"
#include "correlate/oskar_cross_correlate_omp.h"
#include "correlate/oskar_cross_correlate_scalar_cuda.h"
#include "correlate/oskar_cross_correlate_scalar_omp.h"
#include "utility/oskar_device.h"

#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cross_correlate(int num_sources,  const oskar_Jones* jones,
        const oskar_Sky* sky, const oskar_Telescope* tel,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        double gast, double frequency_hz, int offset_out, oskar_Mem* vis,
        int* status)
{
    const oskar_Mem *J, *src_a, *src_b, *src_c, *src_l, *src_m, *src_n;
    const oskar_Mem *src_I, *src_Q, *src_U, *src_V, *x, *y;
    double uv_filter_min, uv_filter_max;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data dimensions. */
    const int num_stations = oskar_telescope_num_stations(tel);
    const int use_extended = oskar_sky_use_extended(sky);

    /* Get bandwidth-smearing terms. */
    frequency_hz = fabs(frequency_hz);
    const double inv_wavelength = frequency_hz / 299792458.0;
    const double channel_bandwidth = oskar_telescope_channel_bandwidth_hz(tel);
    const double frac_bandwidth = channel_bandwidth / frequency_hz;

    /* Get time-average smearing term and Greenwich hour angle. */
    const double time_avg = oskar_telescope_time_average_sec(tel);
    const double gha0 = gast - oskar_telescope_phase_centre_ra_rad(tel);
    const double dec0 = oskar_telescope_phase_centre_dec_rad(tel);

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
    const int location = oskar_sky_mem_location(sky);
    if (oskar_telescope_mem_location(tel) != location ||
            oskar_jones_mem_location(jones) != location ||
            oskar_mem_location(vis) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    const int jones_type = oskar_jones_type(jones);
    const int base_type = oskar_sky_precision(sky);
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

    /* Check the input dimensions. */
    if (oskar_jones_num_sources(jones) < num_sources ||
            (int)oskar_mem_length(u) != num_stations ||
            (int)oskar_mem_length(v) != num_stations ||
            (int)oskar_mem_length(w) != num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get handles to arrays. */
    J = oskar_jones_mem_const(jones);
    src_I = oskar_sky_I_const(sky);
    src_Q = oskar_sky_Q_const(sky);
    src_U = oskar_sky_U_const(sky);
    src_V = oskar_sky_V_const(sky);
    src_l = oskar_sky_l_const(sky);
    src_m = oskar_sky_m_const(sky);
    src_n = oskar_sky_n_const(sky);
    src_a = oskar_sky_gaussian_a_const(sky);
    src_b = oskar_sky_gaussian_b_const(sky);
    src_c = oskar_sky_gaussian_c_const(sky);
    x = oskar_telescope_station_true_x_offset_ecef_metres_const(tel);
    y = oskar_telescope_station_true_y_offset_ecef_metres_const(tel);

    /* Select kernel. */
    if (location == OSKAR_CPU)
    {
        if (use_extended)
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                oskar_cross_correlate_gaussian_omp_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float4c_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_Q, status),
                        oskar_mem_float_const(src_U, status),
                        oskar_mem_float_const(src_V, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(src_a, status),
                        oskar_mem_float_const(src_b, status),
                        oskar_mem_float_const(src_c, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float4c(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                oskar_cross_correlate_gaussian_omp_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double4c_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_Q, status),
                        oskar_mem_double_const(src_U, status),
                        oskar_mem_double_const(src_V, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(src_a, status),
                        oskar_mem_double_const(src_b, status),
                        oskar_mem_double_const(src_c, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double4c(vis, status));
                break;
            case OSKAR_SINGLE_COMPLEX:
                oskar_cross_correlate_scalar_gaussian_omp_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float2_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(src_a, status),
                        oskar_mem_float_const(src_b, status),
                        oskar_mem_float_const(src_c, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float2(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX:
                oskar_cross_correlate_scalar_gaussian_omp_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double2_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(src_a, status),
                        oskar_mem_double_const(src_b, status),
                        oskar_mem_double_const(src_c, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double2(vis, status));
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        else
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                oskar_cross_correlate_point_omp_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float4c_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_Q, status),
                        oskar_mem_float_const(src_U, status),
                        oskar_mem_float_const(src_V, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float4c(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                oskar_cross_correlate_point_omp_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double4c_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_Q, status),
                        oskar_mem_double_const(src_U, status),
                        oskar_mem_double_const(src_V, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double4c(vis, status));
                break;
            case OSKAR_SINGLE_COMPLEX:
                oskar_cross_correlate_scalar_point_omp_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float2_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float2(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX:
                oskar_cross_correlate_scalar_point_omp_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double2_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double2(vis, status));
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (use_extended)
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                oskar_cross_correlate_gaussian_cuda_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float4c_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_Q, status),
                        oskar_mem_float_const(src_U, status),
                        oskar_mem_float_const(src_V, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(src_a, status),
                        oskar_mem_float_const(src_b, status),
                        oskar_mem_float_const(src_c, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float4c(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                oskar_cross_correlate_gaussian_cuda_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double4c_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_Q, status),
                        oskar_mem_double_const(src_U, status),
                        oskar_mem_double_const(src_V, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(src_a, status),
                        oskar_mem_double_const(src_b, status),
                        oskar_mem_double_const(src_c, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double4c(vis, status));
                break;
            case OSKAR_SINGLE_COMPLEX:
                oskar_cross_correlate_scalar_gaussian_cuda_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float2_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(src_a, status),
                        oskar_mem_float_const(src_b, status),
                        oskar_mem_float_const(src_c, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float2(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX:
                oskar_cross_correlate_scalar_gaussian_cuda_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double2_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(src_a, status),
                        oskar_mem_double_const(src_b, status),
                        oskar_mem_double_const(src_c, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double2(vis, status));
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        else
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                oskar_cross_correlate_point_cuda_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float4c_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_Q, status),
                        oskar_mem_float_const(src_U, status),
                        oskar_mem_float_const(src_V, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float4c(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                oskar_cross_correlate_point_cuda_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double4c_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_Q, status),
                        oskar_mem_double_const(src_U, status),
                        oskar_mem_double_const(src_V, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double4c(vis, status));
                break;
            case OSKAR_SINGLE_COMPLEX:
                oskar_cross_correlate_scalar_point_cuda_f(
                        num_sources, num_stations, offset_out,
                        oskar_mem_float2_const(J, status),
                        oskar_mem_float_const(src_I, status),
                        oskar_mem_float_const(src_l, status),
                        oskar_mem_float_const(src_m, status),
                        oskar_mem_float_const(src_n, status),
                        oskar_mem_float_const(u, status),
                        oskar_mem_float_const(v, status),
                        oskar_mem_float_const(w, status),
                        oskar_mem_float_const(x, status),
                        oskar_mem_float_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_float2(vis, status));
                break;
            case OSKAR_DOUBLE_COMPLEX:
                oskar_cross_correlate_scalar_point_cuda_d(
                        num_sources, num_stations, offset_out,
                        oskar_mem_double2_const(J, status),
                        oskar_mem_double_const(src_I, status),
                        oskar_mem_double_const(src_l, status),
                        oskar_mem_double_const(src_m, status),
                        oskar_mem_double_const(src_n, status),
                        oskar_mem_double_const(u, status),
                        oskar_mem_double_const(v, status),
                        oskar_mem_double_const(w, status),
                        oskar_mem_double_const(x, status),
                        oskar_mem_double_const(y, status),
                        uv_filter_min, uv_filter_max, inv_wavelength,
                        frac_bandwidth, time_avg, gha0, dec0,
                        oskar_mem_double2(vis, status));
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        oskar_device_check_error_cuda(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        size_t local_size[] = {128, 1, 1}, global_size[] = {1, 1, 1};
        const int is_dbl = oskar_mem_is_double(vis);
        const int is_matrix = oskar_mem_is_matrix(vis);
        const char* k = 0;
        const float uv_filter_min_f = (float) uv_filter_min;
        const float uv_filter_max_f = (float) uv_filter_max;
        const float inv_wavelength_f = (float) inv_wavelength;
        const float frac_bandwidth_f = (float) frac_bandwidth;
        const float time_avg_f = (float) time_avg;
        const float gha0_f = (float) gha0;
        const float dec0_f = (float) dec0;
        if (use_extended)
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_gaussian_float";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_gaussian_bs_float";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_gaussian_ts_float";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_gaussian_bs_ts_float";
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_gaussian_double";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_gaussian_bs_double";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_gaussian_ts_double";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_gaussian_bs_ts_double";
                break;
            case OSKAR_SINGLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_gaussian_float";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_gaussian_bs_float";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_gaussian_ts_float";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_gaussian_bs_ts_float";
                break;
            case OSKAR_DOUBLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_gaussian_double";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_gaussian_bs_double";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_gaussian_ts_double";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_gaussian_bs_ts_double";
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        else
        {
            switch (oskar_mem_type(vis))
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_point_float";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_point_bs_float";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_point_ts_float";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_point_bs_ts_float";
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_point_double";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_point_bs_double";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_point_ts_double";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_point_bs_ts_double";
                break;
            case OSKAR_SINGLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_point_float";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_point_bs_float";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_point_ts_float";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_point_bs_ts_float";
                break;
            case OSKAR_DOUBLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_point_double";
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                    k = "xcorr_scalar_point_bs_double";
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_point_ts_double";
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                    k = "xcorr_scalar_point_bs_ts_double";
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {INT_SZ, &num_stations},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer_const(src_I)},
                {PTR_SZ, oskar_mem_buffer_const(src_Q)},
                {PTR_SZ, oskar_mem_buffer_const(src_U)},
                {PTR_SZ, oskar_mem_buffer_const(src_V)},
                {PTR_SZ, oskar_mem_buffer_const(src_l)},
                {PTR_SZ, oskar_mem_buffer_const(src_m)},
                {PTR_SZ, oskar_mem_buffer_const(src_n)},
                {PTR_SZ, oskar_mem_buffer_const(src_a)},
                {PTR_SZ, oskar_mem_buffer_const(src_b)},
                {PTR_SZ, oskar_mem_buffer_const(src_c)},
                {PTR_SZ, oskar_mem_buffer_const(u)},
                {PTR_SZ, oskar_mem_buffer_const(v)},
                {PTR_SZ, oskar_mem_buffer_const(w)},
                {PTR_SZ, oskar_mem_buffer_const(x)},
                {PTR_SZ, oskar_mem_buffer_const(y)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&uv_filter_min :
                        (const void*)&uv_filter_min_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&uv_filter_max :
                        (const void*)&uv_filter_max_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&inv_wavelength :
                        (const void*)&inv_wavelength_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&frac_bandwidth :
                        (const void*)&frac_bandwidth_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&time_avg : (const void*)&time_avg_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&gha0 : (const void*)&gha0_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&dec0 : (const void*)&dec0_f},
                {PTR_SZ, oskar_mem_buffer_const(J)},
                {PTR_SZ, oskar_mem_buffer(vis)}
        };
        if (oskar_device_is_cpu(location))
            local_size[0] = 8;
        else if (is_matrix && is_dbl && time_avg != 0.0)
            local_size[0] = 64;
        const size_t arg_size_local[] = {
                local_size[0] * oskar_mem_element_size(oskar_mem_type(vis))
        };
        global_size[0] = num_stations * local_size[0];
        global_size[1] = num_stations;
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args,
                1, arg_size_local, status);
    }
}

#ifdef __cplusplus
}
#endif
