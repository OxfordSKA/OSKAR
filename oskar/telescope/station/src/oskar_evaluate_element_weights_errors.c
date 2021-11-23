/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/define_evaluate_element_weights_errors.h"
#include "telescope/station/oskar_evaluate_element_weights_errors.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_ELEMENT_WEIGHTS_ERR(evaluate_element_weights_errors_f, float, float2)
OSKAR_ELEMENT_WEIGHTS_ERR(evaluate_element_weights_errors_d, double, double2)

void oskar_evaluate_element_weights_errors(int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        unsigned int random_seed, int time_index, int station_id,
        oskar_Mem* errors, int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(errors);
    const int type = oskar_mem_type(errors);
    const int precision = oskar_mem_precision(errors);
    if (oskar_mem_location(gain) != location ||
            oskar_mem_location(gain_error) != location ||
            oskar_mem_location(phase) != location ||
            oskar_mem_location(phase_error) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(gain) != precision ||
            oskar_mem_type(phase) != precision ||
            oskar_mem_type(gain_error) != precision ||
            oskar_mem_type(phase_error) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(errors) < num_elements ||
            (int)oskar_mem_length(gain) < num_elements ||
            (int)oskar_mem_length(gain_error) < num_elements ||
            (int)oskar_mem_length(phase) < num_elements ||
            (int)oskar_mem_length(phase_error) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    oskar_mem_random_gaussian(errors, random_seed, time_index,
            station_id, 0x12345678, 1.0, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            evaluate_element_weights_errors_d(num_elements,
                    oskar_mem_double_const(gain, status),
                    oskar_mem_double_const(gain_error, status),
                    oskar_mem_double_const(phase, status),
                    oskar_mem_double_const(phase_error, status),
                    oskar_mem_double2(errors, status));
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            evaluate_element_weights_errors_f(num_elements,
                    oskar_mem_float_const(gain, status),
                    oskar_mem_float_const(gain_error, status),
                    oskar_mem_float_const(phase, status),
                    oskar_mem_float_const(phase_error, status),
                    oskar_mem_float2(errors, status));
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
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            k = "evaluate_element_weights_errors_double";
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            k = "evaluate_element_weights_errors_float";
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
                {PTR_SZ, oskar_mem_buffer_const(gain)},
                {PTR_SZ, oskar_mem_buffer_const(gain_error)},
                {PTR_SZ, oskar_mem_buffer_const(phase)},
                {PTR_SZ, oskar_mem_buffer_const(phase_error)},
                {PTR_SZ, oskar_mem_buffer(errors)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
