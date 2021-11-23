/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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

void oskar_cross_correlate(
        int source_type,
        int num_sources,
        const oskar_Jones* jones,
        const oskar_Mem* const src_flux[4],
        const oskar_Mem* const src_dir[3],
        const oskar_Mem* const src_ext[3],
        const oskar_Telescope* tel,
        const oskar_Mem* const station_uvw[3],
        double gast,
        double frequency_hz,
        int offset_out,
        oskar_Mem* vis,
        int* status)
{
    const oskar_Mem *J = 0, *x = 0, *y = 0;
    double uv_filter_min = 0.0, uv_filter_max = 0.0;
    double time_avg = 0.0, gha0 = 0.0, dec0 = 0.0;
    if (*status) return;

    /* Get the data dimensions. */
    const int num_stations = oskar_telescope_num_stations(tel);
    const int use_extended = (source_type == 1);

    /* Get bandwidth-smearing terms. */
    frequency_hz = fabs(frequency_hz);
    const double inv_wavelength = frequency_hz / 299792458.0;
    const double channel_bandwidth = oskar_telescope_channel_bandwidth_hz(tel);
    const double frac_bandwidth = channel_bandwidth / frequency_hz;

    /* Get time-average smearing terms.
     * Ignore if drift scanning - this will need to be done differently. */
    if (oskar_telescope_phase_centre_coord_type(tel) != OSKAR_COORDS_AZEL)
    {
        time_avg = oskar_telescope_time_average_sec(tel);
        gha0 = gast - oskar_telescope_phase_centre_longitude_rad(tel);
        dec0 = oskar_telescope_phase_centre_latitude_rad(tel);
    }

    /* Get UV filter parameters in wavelengths. */
    uv_filter_min = oskar_telescope_uv_filter_min(tel);
    uv_filter_max = oskar_telescope_uv_filter_max(tel);
    if (oskar_telescope_uv_filter_units(tel) == OSKAR_METRES)
    {
        uv_filter_min *= inv_wavelength;
        uv_filter_max *= inv_wavelength;
    }
    if (uv_filter_max < 0.0 || uv_filter_max > FLT_MAX)
    {
        uv_filter_max = FLT_MAX;
    }

    /* Check data locations. */
    const int location = oskar_jones_mem_location(jones);
    if (oskar_telescope_mem_location(tel) != location ||
            oskar_mem_location(vis) != location ||
            oskar_mem_location(station_uvw[0]) != location ||
            oskar_mem_location(station_uvw[1]) != location ||
            oskar_mem_location(station_uvw[2]) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    const int jones_type = oskar_jones_type(jones);
    const int base_type = oskar_type_precision(jones_type);
    if (oskar_mem_precision(vis) != base_type ||
            oskar_mem_type(station_uvw[0]) != base_type ||
            oskar_mem_type(station_uvw[1]) != base_type ||
            oskar_mem_type(station_uvw[2]) != base_type)
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
            (int)oskar_mem_length(station_uvw[0]) != num_stations ||
            (int)oskar_mem_length(station_uvw[1]) != num_stations ||
            (int)oskar_mem_length(station_uvw[2]) != num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get handles to arrays. */
    J = oskar_jones_mem_const(jones);
    x = oskar_telescope_station_true_offset_ecef_metres_const(tel, 0);
    y = oskar_telescope_station_true_offset_ecef_metres_const(tel, 1);

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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_flux[1], status),
                        oskar_mem_float_const(src_flux[2], status),
                        oskar_mem_float_const(src_flux[3], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(src_ext[0], status),
                        oskar_mem_float_const(src_ext[1], status),
                        oskar_mem_float_const(src_ext[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_flux[1], status),
                        oskar_mem_double_const(src_flux[2], status),
                        oskar_mem_double_const(src_flux[3], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(src_ext[0], status),
                        oskar_mem_double_const(src_ext[1], status),
                        oskar_mem_double_const(src_ext[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(src_ext[0], status),
                        oskar_mem_float_const(src_ext[1], status),
                        oskar_mem_float_const(src_ext[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(src_ext[0], status),
                        oskar_mem_double_const(src_ext[1], status),
                        oskar_mem_double_const(src_ext[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_flux[1], status),
                        oskar_mem_float_const(src_flux[2], status),
                        oskar_mem_float_const(src_flux[3], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_flux[1], status),
                        oskar_mem_double_const(src_flux[2], status),
                        oskar_mem_double_const(src_flux[3], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_flux[1], status),
                        oskar_mem_float_const(src_flux[2], status),
                        oskar_mem_float_const(src_flux[3], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(src_ext[0], status),
                        oskar_mem_float_const(src_ext[1], status),
                        oskar_mem_float_const(src_ext[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_flux[1], status),
                        oskar_mem_double_const(src_flux[2], status),
                        oskar_mem_double_const(src_flux[3], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(src_ext[0], status),
                        oskar_mem_double_const(src_ext[1], status),
                        oskar_mem_double_const(src_ext[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(src_ext[0], status),
                        oskar_mem_float_const(src_ext[1], status),
                        oskar_mem_float_const(src_ext[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(src_ext[0], status),
                        oskar_mem_double_const(src_ext[1], status),
                        oskar_mem_double_const(src_ext[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_flux[1], status),
                        oskar_mem_float_const(src_flux[2], status),
                        oskar_mem_float_const(src_flux[3], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_flux[1], status),
                        oskar_mem_double_const(src_flux[2], status),
                        oskar_mem_double_const(src_flux[3], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                        oskar_mem_float_const(src_flux[0], status),
                        oskar_mem_float_const(src_dir[0], status),
                        oskar_mem_float_const(src_dir[1], status),
                        oskar_mem_float_const(src_dir[2], status),
                        oskar_mem_float_const(station_uvw[0], status),
                        oskar_mem_float_const(station_uvw[1], status),
                        oskar_mem_float_const(station_uvw[2], status),
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
                        oskar_mem_double_const(src_flux[0], status),
                        oskar_mem_double_const(src_dir[0], status),
                        oskar_mem_double_const(src_dir[1], status),
                        oskar_mem_double_const(src_dir[2], status),
                        oskar_mem_double_const(station_uvw[0], status),
                        oskar_mem_double_const(station_uvw[1], status),
                        oskar_mem_double_const(station_uvw[2], status),
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
                {
                    k = "xcorr_gaussian_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_gaussian_bs_float";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_gaussian_ts_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_gaussian_bs_ts_float";
                }
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_gaussian_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_gaussian_bs_double";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_gaussian_ts_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_gaussian_bs_ts_double";
                }
                break;
            case OSKAR_SINGLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_gaussian_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_gaussian_bs_float";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_gaussian_ts_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_gaussian_bs_ts_float";
                }
                break;
            case OSKAR_DOUBLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_gaussian_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_gaussian_bs_double";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_gaussian_ts_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_gaussian_bs_ts_double";
                }
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
                {
                    k = "xcorr_point_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_point_bs_float";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_point_ts_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_point_bs_ts_float";
                }
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_point_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_point_bs_double";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_point_ts_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_point_bs_ts_double";
                }
                break;
            case OSKAR_SINGLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_point_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_point_bs_float";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_point_ts_float";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_point_bs_ts_float";
                }
                break;
            case OSKAR_DOUBLE_COMPLEX:
                if (frac_bandwidth == 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_point_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg == 0.0)
                {
                    k = "xcorr_scalar_point_bs_double";
                }
                else if (frac_bandwidth == 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_point_ts_double";
                }
                else if (frac_bandwidth != 0.0 && time_avg != 0.0)
                {
                    k = "xcorr_scalar_point_bs_ts_double";
                }
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
                {PTR_SZ, oskar_mem_buffer_const(src_flux[0])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[1])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[2])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[3])},
                {PTR_SZ, oskar_mem_buffer_const(src_dir[0])},
                {PTR_SZ, oskar_mem_buffer_const(src_dir[1])},
                {PTR_SZ, oskar_mem_buffer_const(src_dir[2])},
                {PTR_SZ, oskar_mem_buffer_const(src_ext[0])},
                {PTR_SZ, oskar_mem_buffer_const(src_ext[1])},
                {PTR_SZ, oskar_mem_buffer_const(src_ext[2])},
                {PTR_SZ, oskar_mem_buffer_const(station_uvw[0])},
                {PTR_SZ, oskar_mem_buffer_const(station_uvw[1])},
                {PTR_SZ, oskar_mem_buffer_const(station_uvw[2])},
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
        {
            local_size[0] = 8;
        }
        else if (is_matrix && is_dbl && time_avg != 0.0)
        {
            local_size[0] = 64;
        }
        oskar_device_check_local_size(location, 0, local_size);
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
