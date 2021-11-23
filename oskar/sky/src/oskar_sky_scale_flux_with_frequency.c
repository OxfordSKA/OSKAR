/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/define_sky_scale_flux_with_frequency.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(scale_flux_with_frequency_float, float)
OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(scale_flux_with_frequency_double, double)

void oskar_sky_scale_flux_with_frequency(oskar_Sky* sky, double frequency,
        int* status)
{
    if (*status) return;
    const int type = oskar_sky_precision(sky);
    const int location = oskar_sky_mem_location(sky);
    const int num_sources = oskar_sky_num_sources(sky);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            scale_flux_with_frequency_float(num_sources, frequency,
                    oskar_mem_float(oskar_sky_I(sky), status),
                    oskar_mem_float(oskar_sky_Q(sky), status),
                    oskar_mem_float(oskar_sky_U(sky), status),
                    oskar_mem_float(oskar_sky_V(sky), status),
                    oskar_mem_float(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            scale_flux_with_frequency_double(num_sources, frequency,
                    oskar_mem_double(oskar_sky_I(sky), status),
                    oskar_mem_double(oskar_sky_Q(sky), status),
                    oskar_mem_double(oskar_sky_U(sky), status),
                    oskar_mem_double(oskar_sky_V(sky), status),
                    oskar_mem_double(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const float frequency_f = (float) frequency;
        const char* k = 0;
        const int is_dbl = (type == OSKAR_DOUBLE);
        if (is_dbl)
        {
            k = "scale_flux_with_frequency_double";
        }
        else if (type == OSKAR_SINGLE)
        {
            k = "scale_flux_with_frequency_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&frequency : (const void*)&frequency_f},
                {PTR_SZ, oskar_mem_buffer(oskar_sky_I(sky))},
                {PTR_SZ, oskar_mem_buffer(oskar_sky_Q(sky))},
                {PTR_SZ, oskar_mem_buffer(oskar_sky_U(sky))},
                {PTR_SZ, oskar_mem_buffer(oskar_sky_V(sky))},
                {PTR_SZ, oskar_mem_buffer(oskar_sky_reference_freq_hz(sky))},
                {PTR_SZ, oskar_mem_buffer_const(
                        oskar_sky_spectral_index_const(sky))},
                {PTR_SZ, oskar_mem_buffer_const(
                        oskar_sky_rotation_measure_rad_const(sky))}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
