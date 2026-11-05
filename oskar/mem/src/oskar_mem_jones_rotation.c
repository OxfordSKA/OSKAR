/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/define_mem_jones_rotation.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_MEM_JONES_ROTATION( M_CAT(mem_jones_rotation_, float), float, float4c)
OSKAR_MEM_JONES_ROTATION( M_CAT(mem_jones_rotation_, double), double, double4c)

void oskar_mem_jones_rotation(
        oskar_Mem* mem,
        double angle_rad,
        size_t offset,
        size_t num_elements,
        int* status
)
{
    if (*status) return;
    const int type = oskar_mem_type(mem);
    const int location = oskar_mem_location(mem);
    const unsigned int off = (unsigned int) offset;
    const unsigned int n = (unsigned int) num_elements;
    const double cos_angle = cos(angle_rad);
    const double sin_angle = sin(angle_rad);
    const float cos_angle_f = (float) cos_angle;
    const float sin_angle_f = (float) sin_angle;
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            mem_jones_rotation_double(
                    off, n, cos_angle, sin_angle,
                    oskar_mem_double4c(mem, status)
            );
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            mem_jones_rotation_float(
                    off, n, cos_angle_f, sin_angle_f,
                    oskar_mem_float4c(mem, status)
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
        int is_dbl = 0;
        switch (type)
        {
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "mem_jones_rotation_double";
            is_dbl = 1;
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "mem_jones_rotation_float";
            is_dbl = 0;
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off},
                {INT_SZ, &n},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*) &cos_angle : (const void*) &cos_angle_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*) &sin_angle : (const void*) &sin_angle_f},
                {PTR_SZ, oskar_mem_buffer(mem)},
        };
        oskar_device_launch_kernel(
                k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status
        );
    }
}

#ifdef __cplusplus
}
#endif
