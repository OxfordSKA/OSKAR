/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/define_rotate_virtual_antenna.h"
#include "telescope/station/element/oskar_rotate_virtual_antenna.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_ROTATE_VIRTUAL_ANTENNA(rotate_virtual_antenna_float, float, float2, float4c)
OSKAR_ROTATE_VIRTUAL_ANTENNA(rotate_virtual_antenna_double, double, double2, double4c)

void oskar_rotate_virtual_antenna(int num_elements, int offset,
        double angle_rad, oskar_Mem* beam, int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(beam);
    const int type = oskar_mem_type(beam);
    const double sin_angle = sin(angle_rad);
    const double cos_angle = cos(angle_rad);
    const float sin_angle_f = (float) sin_angle;
    const float cos_angle_f = (float) cos_angle;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            rotate_virtual_antenna_double(
                    num_elements, offset, sin_angle, cos_angle,
                    oskar_mem_double4c(beam, status)
            );
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            rotate_virtual_antenna_float(
                    num_elements, offset, sin_angle_f, cos_angle_f,
                    oskar_mem_float4c(beam, status)
            );
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(beam);
        if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            k = "rotate_virtual_antenna_double";
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            k = "rotate_virtual_antenna_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_elements},
                {INT_SZ, &offset},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_angle :
                        (const void*)&sin_angle_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_angle :
                        (const void*)&cos_angle_f},
                {PTR_SZ, oskar_mem_buffer(beam)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
