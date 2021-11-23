/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_multiply.h"
#include "interferometer/define_jones_apply_station_gains.h"
#include "interferometer/private_jones.h"
#include "interferometer/oskar_jones.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_JONES_APPLY_STATION_GAINS_C(jones_apply_station_gains_complex_float, float2)
OSKAR_JONES_APPLY_STATION_GAINS_C(jones_apply_station_gains_complex_double, double2)
OSKAR_JONES_APPLY_STATION_GAINS_M(jones_apply_station_gains_matrix_float, float4c)
OSKAR_JONES_APPLY_STATION_GAINS_M(jones_apply_station_gains_matrix_double, double4c)

void oskar_jones_apply_station_gains(oskar_Jones* jones,
        oskar_Mem* gains, int* status)
{
    oskar_Mem* mem = 0;
    if (*status) return;
    const int type = oskar_mem_type(gains);
    const int location = oskar_mem_location(gains);
    const int num_sources = oskar_jones_num_sources(jones);
    const int num_stations = oskar_jones_num_stations(jones);
    if ((int)oskar_mem_length(gains) < num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    mem = oskar_jones_mem(jones);
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX:
            jones_apply_station_gains_complex_float(
                    num_sources, num_stations,
                    oskar_mem_float2_const(gains, status),
                    oskar_mem_float2(mem, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            jones_apply_station_gains_matrix_float(
                    num_sources, num_stations,
                    oskar_mem_float4c_const(gains, status),
                    oskar_mem_float4c(mem, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            jones_apply_station_gains_complex_double(
                    num_sources, num_stations,
                    oskar_mem_double2_const(gains, status),
                    oskar_mem_double2(mem, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            jones_apply_station_gains_matrix_double(
                    num_sources, num_stations,
                    oskar_mem_double4c_const(gains, status),
                    oskar_mem_double4c(mem, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {64, 4, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX:
            k = "jones_apply_station_gains_complex_float";
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "jones_apply_station_gains_matrix_float";
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "jones_apply_station_gains_complex_double";
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "jones_apply_station_gains_matrix_double";
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        global_size[1] = oskar_device_global_size(
                (size_t) num_stations, local_size[1]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {INT_SZ, &num_stations},
                {PTR_SZ, oskar_mem_buffer_const(gains)},
                {PTR_SZ, oskar_mem_buffer(mem)}
        };
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
