/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "correlate/define_auto_correlate.h"
#include "correlate/define_correlate_utils.h"
#include "correlate/oskar_auto_correlate.h"
#include "math/define_multiply.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_ACORR_CPU(acorr_float, float, float2, float4c)
OSKAR_ACORR_CPU(acorr_double, double, double2, double4c)
OSKAR_ACORR_SCALAR_CPU(acorr_scalar_float, float, float2)
OSKAR_ACORR_SCALAR_CPU(acorr_scalar_double, double, double2)

void oskar_auto_correlate(
        int num_sources,
        const oskar_Jones* jones,
        const oskar_Mem* const src_flux[4],
        int offset_out,
        oskar_Mem* vis,
        int* status)
{
    if (*status) return;
    const oskar_Mem* jones_ = oskar_jones_mem_const(jones);
    const int num_stations = oskar_jones_num_stations(jones);
    const int location = oskar_mem_location(jones_);
    if (oskar_mem_location(vis) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(jones_) != oskar_mem_type(vis))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_jones_num_sources(jones) < num_sources)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(vis))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            acorr_float(num_sources, num_stations, offset_out,
                    oskar_mem_float_const(src_flux[0], status),
                    oskar_mem_float_const(src_flux[1], status),
                    oskar_mem_float_const(src_flux[2], status),
                    oskar_mem_float_const(src_flux[3], status),
                    oskar_mem_float4c_const(jones_, status),
                    oskar_mem_float4c(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            acorr_double(num_sources, num_stations, offset_out,
                    oskar_mem_double_const(src_flux[0], status),
                    oskar_mem_double_const(src_flux[1], status),
                    oskar_mem_double_const(src_flux[2], status),
                    oskar_mem_double_const(src_flux[3], status),
                    oskar_mem_double4c_const(jones_, status),
                    oskar_mem_double4c(vis, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            acorr_scalar_float(num_sources, num_stations, offset_out,
                    oskar_mem_float_const(src_flux[0], status),
                    oskar_mem_float_const(src_flux[1], status),
                    oskar_mem_float_const(src_flux[2], status),
                    oskar_mem_float_const(src_flux[3], status),
                    oskar_mem_float2_const(jones_, status),
                    oskar_mem_float2(vis, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            acorr_scalar_double(num_sources, num_stations, offset_out,
                    oskar_mem_double_const(src_flux[0], status),
                    oskar_mem_double_const(src_flux[1], status),
                    oskar_mem_double_const(src_flux[2], status),
                    oskar_mem_double_const(src_flux[3], status),
                    oskar_mem_double2_const(jones_, status),
                    oskar_mem_double2(vis, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {128, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (oskar_mem_type(vis))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX: k = "acorr_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX: k = "acorr_double"; break;
        case OSKAR_SINGLE_COMPLEX:        k = "acorr_scalar_float"; break;
        case OSKAR_DOUBLE_COMPLEX:        k = "acorr_scalar_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = num_stations * local_size[0];
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {INT_SZ, &num_stations},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[0])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[1])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[2])},
                {PTR_SZ, oskar_mem_buffer_const(src_flux[3])},
                {PTR_SZ, oskar_mem_buffer_const(jones_)},
                {PTR_SZ, oskar_mem_buffer(vis)}
        };
        const size_t arg_size_local[] = {
                local_size[0] * oskar_mem_element_size(oskar_mem_type(vis))
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args,
                1, arg_size_local, status);
    }
}

#ifdef __cplusplus
}
#endif
