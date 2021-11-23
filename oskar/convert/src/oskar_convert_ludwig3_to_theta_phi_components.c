/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_ludwig3_to_theta_phi_components.h"
#include "convert/oskar_convert_ludwig3_to_theta_phi_components.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI(convert_ludwig3_to_theta_phi_float, float, float2)
OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI(convert_ludwig3_to_theta_phi_double, double, double2)

void oskar_convert_ludwig3_to_theta_phi_components(
        int num_points, const oskar_Mem* phi, int stride, int offset,
        oskar_Mem* vec, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(phi);
    const int location = oskar_mem_location(phi);
    const int off_h = offset, off_v = offset + 1;
    if (!oskar_mem_is_matrix(vec))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_ludwig3_to_theta_phi_float(
                    num_points,
                    oskar_mem_float_const(phi, status),
                    stride, off_h, off_v,
                    oskar_mem_float2(vec, status),
                    oskar_mem_float2(vec, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_ludwig3_to_theta_phi_double(
                    num_points,
                    oskar_mem_double_const(phi, status),
                    stride, off_h, off_v,
                    oskar_mem_double2(vec, status),
                    oskar_mem_double2(vec, status));
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
        if (type == OSKAR_SINGLE)
        {
            k = "convert_ludwig3_to_theta_phi_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_ludwig3_to_theta_phi_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(phi)},
                {INT_SZ, &stride},
                {INT_SZ, &off_h},
                {INT_SZ, &off_v},
                {PTR_SZ, oskar_mem_buffer(vec)},
                {PTR_SZ, oskar_mem_buffer(vec)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
