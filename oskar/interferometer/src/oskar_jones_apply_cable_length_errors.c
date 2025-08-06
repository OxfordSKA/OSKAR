/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_multiply.h"
#include "interferometer/define_jones_apply_cable_length_errors.h"
#include "interferometer/private_jones.h"
#include "interferometer/oskar_jones.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_C(jones_apply_cable_length_errors_complex_float, float, float2)
OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_C(jones_apply_cable_length_errors_complex_double, double, double2)
OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_M(jones_apply_cable_length_errors_matrix_float, float, float4c)
OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_M(jones_apply_cable_length_errors_matrix_double, double, double4c)

void oskar_jones_apply_cable_length_errors(
        oskar_Jones* jones,
        double frequency_hz,
        const oskar_Mem* errors_x,
        const oskar_Mem* errors_y,
        int* status
)
{
    oskar_Mem* mem = 0;
    if (*status) return;
    const int apply_x = (errors_x != 0);
    const int apply_y = (errors_y != 0);
    if (!apply_x && !apply_y) return;
    const int type = oskar_mem_type(oskar_jones_mem(jones));
    const int location = oskar_mem_location(oskar_jones_mem(jones));
    const int num_sources = oskar_jones_num_sources(jones);
    const int num_stations = oskar_jones_num_stations(jones);
    const double wavenumber = 2 * M_PI * frequency_hz / 299792458.0;
    const float wavenumber_f = (float) wavenumber;
    if ((errors_x && ((int) oskar_mem_length(errors_x) < num_stations)) ||
            (errors_y && ((int) oskar_mem_length(errors_y) < num_stations)))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;           /* LCOV_EXCL_LINE */
        return;                                           /* LCOV_EXCL_LINE */
    }
    mem = oskar_jones_mem(jones);
    if (location == OSKAR_CPU)
    {
        const void* ptr_x = errors_x ? oskar_mem_void_const(errors_x) : 0;
        const void* ptr_y = errors_y ? oskar_mem_void_const(errors_y) : 0;
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX:
            jones_apply_cable_length_errors_complex_float(
                    num_sources, num_stations, wavenumber_f, apply_x, apply_y,
                    (const float*) ptr_x, (const float*) ptr_y,
                    oskar_mem_float2(mem, status)
            );
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            jones_apply_cable_length_errors_matrix_float(
                    num_sources, num_stations, wavenumber_f, apply_x, apply_y,
                    (const float*) ptr_x, (const float*) ptr_y,
                    oskar_mem_float4c(mem, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX:
            jones_apply_cable_length_errors_complex_double(
                    num_sources, num_stations, wavenumber, apply_x, apply_y,
                    (const double*) ptr_x, (const double*) ptr_y,
                    oskar_mem_double2(mem, status)
            );
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            jones_apply_cable_length_errors_matrix_double(
                    num_sources, num_stations, wavenumber, apply_x, apply_y,
                    (const double*) ptr_x, (const double*) ptr_y,
                    oskar_mem_double4c(mem, status)
            );
            break;
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
    }
    else
    {
        size_t local_size[] = {64, 4, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const void* nullp = 0;
        int is_dbl = 0;
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX:
            k = "jones_apply_cable_length_errors_complex_float";
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "jones_apply_cable_length_errors_matrix_float";
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "jones_apply_cable_length_errors_complex_double";
            is_dbl = 1;
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "jones_apply_cable_length_errors_matrix_double";
            is_dbl = 1;
            break;
        default:                                          /* LCOV_EXCL_LINE */
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]
        );
        global_size[1] = oskar_device_global_size(
                (size_t) num_stations, local_size[1]
        );
        const oskar_Arg args[] =
        {
            {INT_SZ, &num_sources},
            {INT_SZ, &num_stations},
            {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                    (const void*) &wavenumber : (const void*) &wavenumber_f
            },
            {INT_SZ, &apply_x},
            {INT_SZ, &apply_y},
            {PTR_SZ, apply_x ? oskar_mem_buffer_const(errors_x) : &nullp},
            {PTR_SZ, apply_y ? oskar_mem_buffer_const(errors_y) : &nullp},
            {PTR_SZ, oskar_mem_buffer(mem)}
        };
        oskar_device_launch_kernel(
                k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status
        );
    }
}

#ifdef __cplusplus
}
#endif
