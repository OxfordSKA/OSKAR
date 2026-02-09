/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "oskar/telescope/station/element/oskar_evaluate_spherical_wave_sum_feko_norm_polynomial.h"
#include "oskar/telescope/station/element/define_evaluate_spherical_wave_feko_norm_polynomial.h"
#include "oskar/log/oskar_log.h"
#include "oskar/math/define_legendre_polynomial_norm.h"
#include "oskar/math/define_multiply.h"
#include "oskar/utility/oskar_device.h"
#include "oskar/utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_IPOW(ipow_float, float, float2)
OSKAR_IPOW(ipow_double, double, double2)

OSKAR_LEGENDRE_NORM(legendre_norm_float, float)
OSKAR_LEGENDRE_NORM(legendre_norm_double, double)

OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_NORM(evaluate_spherical_wave_sum_feko_norm_float, float, float2, float4c)
OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_NORM(evaluate_spherical_wave_sum_feko_norm_double, double, double2, double4c)


void oskar_evaluate_spherical_wave_sum_feko_norm_polynomial(
        int num_points,
        const oskar_Mem* theta_rad,
        const oskar_Mem* phi_x_rad,
        const oskar_Mem* phi_y_rad,
        int n_max,
        const oskar_Mem* root_n,
        int use_ticra_convention,
        const oskar_Mem* coeffs,
        oskar_Mem* workspace,
        int swap_xy,
        int offset_out,
        oskar_Mem* pattern,
        int* status
)
{
    if (*status) return;
    const int location = oskar_mem_location(pattern);
    const double coeff_scale = use_ticra_convention ? sqrt(8. * M_PI) : 1.;
    const float coeff_scale_f = (float) coeff_scale;
    const int coeff_required = (n_max + 1) * (n_max + 1) - 1;
    if (oskar_mem_length(coeffs) < (size_t) coeff_required)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;           /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    if (oskar_mem_location(theta_rad) != location ||
            oskar_mem_location(phi_x_rad) != location ||
            oskar_mem_location(phi_y_rad) != location ||
            oskar_mem_location(coeffs) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;            /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    if (!oskar_mem_is_complex(pattern))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_feko_norm_float(
                    num_points,
                    oskar_mem_float_const(theta_rad, status),
                    oskar_mem_float_const(phi_x_rad, status),
                    oskar_mem_float_const(phi_y_rad, status),
                    n_max,
                    oskar_mem_float_const(root_n, status),
                    use_ticra_convention,
                    oskar_mem_float2_const(coeffs, status),
                    coeff_scale_f,
                    oskar_mem_float(workspace, status),
                    swap_xy,
                    offset_out,
                    oskar_mem_float4c(pattern, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_feko_norm_double(
                    num_points,
                    oskar_mem_double_const(theta_rad, status),
                    oskar_mem_double_const(phi_x_rad, status),
                    oskar_mem_double_const(phi_y_rad, status),
                    n_max,
                    oskar_mem_double_const(root_n, status),
                    use_ticra_convention,
                    oskar_mem_double2_const(coeffs, status),
                    coeff_scale,
                    oskar_mem_double(workspace, status),
                    swap_xy,
                    offset_out,
                    oskar_mem_double4c(pattern, status)
            );
            break;
        case OSKAR_SINGLE_COMPLEX:                        /* LCOV_EXCL_LINE */
        case OSKAR_DOUBLE_COMPLEX:                        /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Spherical wave patterns cannot be used in scalar mode"
            );
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        }
    }
    else
    {
        int is_dbl = 1;
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            is_dbl = 0;
            k = "evaluate_spherical_wave_sum_feko_norm_float";
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            is_dbl = 1;
            k = "evaluate_spherical_wave_sum_feko_norm_double";
            break;
        case OSKAR_SINGLE_COMPLEX:                        /* LCOV_EXCL_LINE */
        case OSKAR_DOUBLE_COMPLEX:                        /* LCOV_EXCL_LINE */
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0, "Spherical wave patterns cannot be used in scalar mode"
            );
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]
        );
        const oskar_Arg arg[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(theta_rad)},
                {PTR_SZ, oskar_mem_buffer_const(phi_x_rad)},
                {PTR_SZ, oskar_mem_buffer_const(phi_y_rad)},
                {INT_SZ, &n_max},
                {PTR_SZ, oskar_mem_buffer_const(root_n)},
                {INT_SZ, &use_ticra_convention},
                {PTR_SZ, oskar_mem_buffer_const(coeffs)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*) &coeff_scale :
                        (const void*) &coeff_scale_f},
                {PTR_SZ, oskar_mem_buffer_const(workspace)},
                {INT_SZ, &swap_xy},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(pattern)}
        };
        oskar_device_launch_kernel(
                k, location, 1, local_size, global_size,
                sizeof(arg) / sizeof(oskar_Arg), arg, 0, 0, status
        );
    }
}

#ifdef __cplusplus
}
#endif
