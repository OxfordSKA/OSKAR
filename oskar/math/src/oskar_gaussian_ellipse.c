/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_gaussian_ellipse.h"
#include "math/oskar_gaussian_ellipse.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_GAUSSIAN_ELLIPSE_MATRIX(gaussian_ellipse_matrix_f, float, float4c)
OSKAR_GAUSSIAN_ELLIPSE_MATRIX(gaussian_ellipse_matrix_d, double, double4c)

void oskar_gaussian_ellipse(
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        double x_a,
        double x_b,
        double x_c,
        double y_a,
        double y_b,
        double y_c,
        oskar_Mem* out,
        int* status
)
{
    if (*status) return;
    const int location = oskar_mem_location(out);
    const float x_a_f = (float)x_a;
    const float x_b_f = (float)x_b;
    const float x_c_f = (float)x_c;
    const float y_a_f = (float)y_a;
    const float y_b_f = (float)y_b;
    const float y_c_f = (float)y_c;
    if (oskar_mem_precision(out) != oskar_mem_type(l) ||
            oskar_mem_precision(out) != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(l) || location != oskar_mem_location(m))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(l) < num_points ||
            (int)oskar_mem_length(m) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    oskar_mem_ensure(out, num_points, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(out))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            gaussian_ellipse_matrix_f(num_points,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    x_a_f, x_b_f, x_c_f, y_a_f, y_b_f, y_c_f,
                    oskar_mem_float4c(out, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            gaussian_ellipse_matrix_d(num_points,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    x_a, x_b, x_c, y_a, y_b, y_c,
                    oskar_mem_double4c(out, status));
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
        const int is_dbl = oskar_mem_is_double(out);
        switch (oskar_mem_type(out))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "gaussian_ellipse_matrix_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "gaussian_ellipse_matrix_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&x_a : (const void*)&x_a_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&x_b : (const void*)&x_b_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&x_c : (const void*)&x_c_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&y_a : (const void*)&y_a_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&y_b : (const void*)&y_b_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&y_c : (const void*)&y_c_f},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
