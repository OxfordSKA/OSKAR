/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/define_evaluate_element_weights_dft.h"
#include "telescope/station/oskar_evaluate_element_weights_dft.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_ELEMENT_WEIGHTS_DFT(evaluate_element_weights_dft_float, float, float2)
OSKAR_ELEMENT_WEIGHTS_DFT(evaluate_element_weights_dft_double, double, double2)

void oskar_evaluate_element_weights_dft(int num_elements,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* cable_length_error, double wavenumber,
        double x_beam, double y_beam, double z_beam, oskar_Mem* weights,
        int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(weights);
    const int type = oskar_mem_type(weights);
    const int precision = oskar_mem_precision(weights);
    if (oskar_mem_location(x) != location ||
            oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(x) != precision || oskar_mem_type(y) != precision ||
            oskar_mem_type(z) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(weights) < num_elements ||
            (int)oskar_mem_length(x) < num_elements ||
            (int)oskar_mem_length(y) < num_elements ||
            (int)oskar_mem_length(z) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            evaluate_element_weights_dft_double(num_elements,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    oskar_mem_double_const(cable_length_error, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_double2(weights, status));
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            evaluate_element_weights_dft_float(num_elements,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    oskar_mem_float_const(cable_length_error, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_float2(weights, status));
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
        const int is_dbl = oskar_mem_is_double(weights);
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            k = "evaluate_element_weights_dft_double";
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            k = "evaluate_element_weights_dft_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        const float w = (float) wavenumber;
        const float x1 = (float) x_beam;
        const float y1 = (float) y_beam;
        const float z1 = (float) z_beam;
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_elements},
                {PTR_SZ, oskar_mem_buffer_const(x)},
                {PTR_SZ, oskar_mem_buffer_const(y)},
                {PTR_SZ, oskar_mem_buffer_const(z)},
                {PTR_SZ, oskar_mem_buffer_const(cable_length_error)},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&wavenumber : (const void*)&w},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&x_beam : (const void*)&x1},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&y_beam : (const void*)&y1},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&z_beam : (const void*)&z1},
                {PTR_SZ, oskar_mem_buffer(weights)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
