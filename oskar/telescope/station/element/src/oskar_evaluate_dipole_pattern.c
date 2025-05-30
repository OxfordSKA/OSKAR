/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/define_evaluate_dipole_pattern.h"
#include "telescope/station/element/oskar_evaluate_dipole_pattern.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EVALUATE_DIPOLE_PATTERN(evaluate_dipole_pattern_f, float, float2, float4c)
OSKAR_EVALUATE_DIPOLE_PATTERN(evaluate_dipole_pattern_d, double, double2, double4c)
OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(evaluate_dipole_pattern_scalar_f, float, float2)
OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(evaluate_dipole_pattern_scalar_d, double, double2)

void oskar_evaluate_dipole_pattern(
        int num_points,
        const oskar_Mem* theta,
        const oskar_Mem* phi_x,
        const oskar_Mem* phi_y,
        double freq_hz,
        double dipole_length_m,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
)
{
    if (*status) return;
    const int precision = oskar_mem_precision(pattern);
    const int location = oskar_mem_location(pattern);
    const double kL = dipole_length_m * (M_PI * freq_hz / 299792458.0);
    const double cos_kL = cos(kL);
    const float kL_f = (float) kL;
    const float cos_kL_f = (float) cos_kL;
    if (oskar_mem_location(theta) != location ||
            oskar_mem_location(phi_x) != location ||
            oskar_mem_location(phi_y) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(theta) != precision ||
            oskar_mem_type(phi_x) != precision ||
            oskar_mem_type(phi_y) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_dipole_pattern_f(
                    num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi_x, status),
                    oskar_mem_float_const(phi_y, status),
                    kL_f,
                    cos_kL_f,
                    swap_xy,
                    offset_out,
                    oskar_mem_float4c(pattern, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_dipole_pattern_d(
                    num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi_x, status),
                    oskar_mem_double_const(phi_y, status),
                    kL,
                    cos_kL,
                    swap_xy,
                    offset_out,
                    oskar_mem_double4c(pattern, status)
            );
            break;
        case OSKAR_SINGLE_COMPLEX:
            evaluate_dipole_pattern_scalar_f(
                    num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi_x, status),
                    oskar_mem_float_const(phi_y, status),
                    kL_f,
                    cos_kL_f,
                    swap_xy,
                    offset_out,
                    oskar_mem_float2(pattern, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX:
            evaluate_dipole_pattern_scalar_d(
                    num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi_x, status),
                    oskar_mem_double_const(phi_y, status),
                    kL,
                    cos_kL,
                    swap_xy,
                    offset_out,
                    oskar_mem_double2(pattern, status)
            );
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(pattern);
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_dipole_pattern_float";
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_dipole_pattern_double";
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = "evaluate_dipole_pattern_scalar_float";
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "evaluate_dipole_pattern_scalar_double";
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]
        );
        const oskar_Arg args[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(theta)},
                {PTR_SZ, oskar_mem_buffer_const(phi_x)},
                {PTR_SZ, oskar_mem_buffer_const(phi_y)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&kL : (const void*)&kL_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_kL : (const void*)&cos_kL_f},
                {INT_SZ, &swap_xy},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(pattern)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
