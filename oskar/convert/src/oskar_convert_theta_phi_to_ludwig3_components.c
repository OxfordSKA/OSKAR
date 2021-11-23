/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_theta_phi_to_ludwig3_components.h"
#include "convert/oskar_convert_theta_phi_to_ludwig3_components.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3(convert_theta_phi_to_ludwig3_float, float, float2, float4c)
OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3(convert_theta_phi_to_ludwig3_double, double, double2, double4c)

void oskar_convert_theta_phi_to_ludwig3_components(
        int num_points, const oskar_Mem* phi_x, const oskar_Mem* phi_y,
        int swap_xy, int offset, oskar_Mem* jones, int* status)
{
    if (*status) return;
    (void)phi_y;
    const int type = oskar_mem_precision(jones);
    const int location = oskar_mem_location(jones);
    if (!oskar_mem_is_matrix(jones))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_theta_phi_to_ludwig3_float(num_points,
                    oskar_mem_float_const(phi_x, status),
                    swap_xy, offset,
                    oskar_mem_float4c(jones, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_theta_phi_to_ludwig3_double(num_points,
                    oskar_mem_double_const(phi_x, status),
                    swap_xy, offset,
                    oskar_mem_double4c(jones, status));
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
            k = "convert_theta_phi_to_ludwig3_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_theta_phi_to_ludwig3_double";
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
                {PTR_SZ, oskar_mem_buffer_const(phi_x)},
                {INT_SZ, &swap_xy},
                {INT_SZ, &offset},
                {PTR_SZ, oskar_mem_buffer(jones)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
