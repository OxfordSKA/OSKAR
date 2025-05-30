/*
 * Copyright (c) 2023-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/oskar_evaluate_spherical_wave_sum_galileo.h"
#include "telescope/station/element/define_evaluate_spherical_wave_galileo.h"
#include "log/oskar_log.h"
#include "math/define_legendre_polynomial.h"
#include "math/define_multiply.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_GALILEO(evaluate_spherical_wave_sum_galileo_float, float, float2, float4c)
OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_GALILEO(evaluate_spherical_wave_sum_galileo_double, double, double2, double4c)

void oskar_evaluate_spherical_wave_sum_galileo(
        int num_points,
        const oskar_Mem* theta,
        const oskar_Mem* phi_x,
        const oskar_Mem* phi_y,
        int l_max,
        const oskar_Mem* alpha,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
)
{
    if (*status) return;
    const int location = oskar_mem_location(pattern);
    const int coeff_required = (l_max + 1) * (l_max + 1) - 1;
    if (oskar_mem_length(alpha) < (size_t) coeff_required)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_galileo_float(
                    num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi_x, status),
                    oskar_mem_float_const(phi_y, status),
                    l_max,
                    oskar_mem_float4c_const(alpha, status),
                    swap_xy,
                    offset_out,
                    oskar_mem_float4c(pattern, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_galileo_double(
                    num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi_x, status),
                    oskar_mem_double_const(phi_y, status),
                    l_max,
                    oskar_mem_double4c_const(alpha, status),
                    swap_xy,
                    offset_out,
                    oskar_mem_double4c(pattern, status)
            );
            break;
        case OSKAR_SINGLE_COMPLEX:
        case OSKAR_DOUBLE_COMPLEX:
            oskar_log_error(
                    0, "Spherical wave patterns cannot be used in scalar mode"
            );
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_spherical_wave_sum_galileo_float";
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_spherical_wave_sum_galileo_double";
            break;
        case OSKAR_SINGLE_COMPLEX:
        case OSKAR_DOUBLE_COMPLEX:
            oskar_log_error(
                    0, "Spherical wave patterns cannot be used in scalar mode"
            );
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg arg[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(theta)},
                {PTR_SZ, oskar_mem_buffer_const(phi_x)},
                {PTR_SZ, oskar_mem_buffer_const(phi_y)},
                {INT_SZ, &l_max},
                {PTR_SZ, oskar_mem_buffer_const(alpha)},
                {INT_SZ, &swap_xy},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(pattern)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(arg) / sizeof(oskar_Arg), arg, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
