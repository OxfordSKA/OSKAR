/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/oskar_jones.h"
#include "interferometer/define_evaluate_jones_R.h"
#include "interferometer/oskar_evaluate_jones_R.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_JONES_R(evaluate_jones_R_float, float, float4c)
OSKAR_JONES_R(evaluate_jones_R_double, double, double4c)

/* NOLINTNEXTLINE(readability-identifier-naming) */
void oskar_evaluate_jones_R(
        oskar_Jones* R,
        int num_sources,
        const oskar_Mem* ra_rad,
        const oskar_Mem* dec_rad,
        const oskar_Telescope* telescope,
        double gast,
        int* status)
{
    int i = 0;
    if (*status) return;
    const int type = oskar_jones_type(R);
    const int precision = oskar_type_precision(type);
    const int location = oskar_jones_mem_location(R);
    const int num_stations = oskar_jones_num_stations(R);
    const int stride = oskar_jones_num_sources(R);
    const int n = (oskar_telescope_allow_station_beam_duplication(telescope) ?
            1 : num_stations);
    if (num_sources > (int)oskar_mem_length(ra_rad) ||
            num_sources > (int)oskar_mem_length(dec_rad) ||
            num_sources > oskar_jones_num_sources(R) ||
            num_stations != oskar_telescope_num_stations(telescope))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(ra_rad) ||
            location != oskar_mem_location(dec_rad))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (precision != oskar_mem_precision(ra_rad) ||
            precision != oskar_mem_precision(dec_rad))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    for (i = 0; i < n; ++i)
    {
        const oskar_Station* st = oskar_telescope_station_const(telescope, i);
        const int offset_out = i * stride;
        const double latitude =   oskar_station_lat_rad(st);
        const double lst = gast + oskar_station_lon_rad(st);
        const double cos_lat = cos(latitude);
        const double sin_lat = sin(latitude);
        const float lst_f = (float)lst;
        const float cos_lat_f = (float)cos_lat;
        const float sin_lat_f = (float)sin_lat;
        if (location == OSKAR_CPU)
        {
            if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                evaluate_jones_R_float(
                        num_sources,
                        oskar_mem_float_const(ra_rad, status),
                        oskar_mem_float_const(dec_rad, status),
                        cos_lat_f, sin_lat_f, lst_f, offset_out,
                        oskar_mem_float4c(oskar_jones_mem(R), status));
            }
            else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                evaluate_jones_R_double(
                        num_sources,
                        oskar_mem_double_const(ra_rad, status),
                        oskar_mem_double_const(dec_rad, status),
                        cos_lat, sin_lat, lst, offset_out,
                        oskar_mem_double4c(oskar_jones_mem(R), status));
            }
            else
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
        }
        else
        {
            size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
            const char* k = 0;
            const int is_dbl = (precision == OSKAR_DOUBLE);
            switch (type)
            {
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                k = "evaluate_jones_R_float"; break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                k = "evaluate_jones_R_double"; break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return;
            }
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(
                    (size_t) num_sources, local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &num_sources},
                    {PTR_SZ, oskar_mem_buffer_const(ra_rad)},
                    {PTR_SZ, oskar_mem_buffer_const(dec_rad)},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&cos_lat : (const void*)&cos_lat_f},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&sin_lat : (const void*)&sin_lat_f},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&lst : (const void*)&lst_f},
                    {INT_SZ, &offset_out},
                    {PTR_SZ, oskar_mem_buffer(oskar_jones_mem(R))}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
    }

    /* Copy data for station 0 to stations 1 to n, if using a common sky. */
    if (oskar_telescope_allow_station_beam_duplication(telescope))
    {
        for (i = 1; i < num_stations; ++i)
        {
            oskar_mem_copy_contents(
                    oskar_jones_mem(R), oskar_jones_mem(R),
                    (size_t)(i * stride), 0,
                    (size_t)stride, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
