/*
 * Copyright (c) 2011-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"
#include "sky/define_sky_scale_flux_with_frequency.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_SKY_CALC_STOKES_I_FLUX(calc_stokes_i_flux_float, float)
OSKAR_SKY_CALC_STOKES_I_FLUX(calc_stokes_i_flux_double, double)
OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(scale_flux_with_frequency_float, float)
OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY(scale_flux_with_frequency_double, double)


void oskar_sky_scale_flux_with_frequency(
        oskar_Sky* sky,
        double freq_hz,
        int* status
)
{
    if (*status) return;
    const int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    const int location = oskar_sky_int(sky, OSKAR_SKY_MEM_LOCATION);
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    const int capacity = oskar_sky_int(sky, OSKAR_SKY_CAPACITY);

    /* Get start indices of relevant sky model columns. */
    const int in_i = oskar_sky_first_column(sky, OSKAR_SKY_I_JY);
    const int in_q = oskar_sky_first_column(sky, OSKAR_SKY_Q_JY);
    const int in_u = oskar_sky_first_column(sky, OSKAR_SKY_U_JY);
    const int in_v = oskar_sky_first_column(sky, OSKAR_SKY_V_JY);
    const int ref_hz = oskar_sky_first_column(sky, OSKAR_SKY_REF_HZ);
    const int spx = oskar_sky_first_column(sky, OSKAR_SKY_SPEC_IDX);
    const int lin_si = oskar_sky_first_column(sky, OSKAR_SKY_LIN_SI);
    const int rm = oskar_sky_first_column(sky, OSKAR_SKY_RM_RAD);
    const int pola = oskar_sky_first_column(sky, OSKAR_SKY_POLA_RAD);
    const int polf = oskar_sky_first_column(sky, OSKAR_SKY_POLF);
    const int refw = oskar_sky_first_column(sky, OSKAR_SKY_REF_WAVE_M);
    const int sp_curv = oskar_sky_first_column(sky, OSKAR_SKY_SPEC_CURV);
    const int width_hz = oskar_sky_first_column(sky, OSKAR_SKY_LINE_WIDTH_HZ);
    const int inc_hz = oskar_sky_first_column(sky, OSKAR_SKY_INC_HZ);

    /* Get pointers to the output scratch columns. */
    oskar_Mem* out_i = oskar_sky_column(
            sky, OSKAR_SKY_SCRATCH_I_JY, 0, status
    );
    oskar_Mem* out_q = oskar_sky_column(
            sky, OSKAR_SKY_SCRATCH_Q_JY, 0, status
    );
    oskar_Mem* out_u = oskar_sky_column(
            sky, OSKAR_SKY_SCRATCH_U_JY, 0, status
    );
    oskar_Mem* out_v = oskar_sky_column(
            sky, OSKAR_SKY_SCRATCH_V_JY, 0, status
    );
    if (in_i < 0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        oskar_log_error(0, "No Stokes I column in sky model.");
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            scale_flux_with_frequency_float(
                    num_sources, capacity, freq_hz,
                    oskar_mem_float_const(sky->table, status),
                    oskar_mem_int_const(sky->num_valid_columns, status),
                    in_i, in_q, in_u, in_v, ref_hz, lin_si, spx,
                    rm, polf, pola, refw, sp_curv, width_hz, inc_hz,
                    oskar_mem_float(out_i, status),
                    oskar_mem_float(out_q, status),
                    oskar_mem_float(out_u, status),
                    oskar_mem_float(out_v, status)
            );
        }
        else if (type == OSKAR_DOUBLE)
        {
            scale_flux_with_frequency_double(
                    num_sources, capacity, freq_hz,
                    oskar_mem_double_const(sky->table, status),
                    oskar_mem_int_const(sky->num_valid_columns, status),
                    in_i, in_q, in_u, in_v, ref_hz, lin_si, spx,
                    rm, polf, pola, refw, sp_curv, width_hz, inc_hz,
                    oskar_mem_double(out_i, status),
                    oskar_mem_double(out_q, status),
                    oskar_mem_double(out_u, status),
                    oskar_mem_double(out_v, status)
            );
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const float freq_hz_f = (float) freq_hz;
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
            *status = OSKAR_ERR_BAD_DATA_TYPE;            /* LCOV_EXCL_LINE */
            return;                                       /* LCOV_EXCL_LINE */
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]
        );
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {INT_SZ, &capacity},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&freq_hz : (const void*)&freq_hz_f},
                {PTR_SZ, oskar_mem_buffer_const(sky->table)},
                {PTR_SZ, oskar_mem_buffer_const(sky->num_valid_columns)},
                {INT_SZ, &in_i},
                {INT_SZ, &in_q},
                {INT_SZ, &in_u},
                {INT_SZ, &in_v},
                {INT_SZ, &ref_hz},
                {INT_SZ, &lin_si},
                {INT_SZ, &spx},
                {INT_SZ, &rm},
                {INT_SZ, &polf},
                {INT_SZ, &pola},
                {INT_SZ, &refw},
                {INT_SZ, &sp_curv},
                {INT_SZ, &width_hz},
                {INT_SZ, &inc_hz},
                {PTR_SZ, oskar_mem_buffer(out_i)},
                {PTR_SZ, oskar_mem_buffer(out_q)},
                {PTR_SZ, oskar_mem_buffer(out_u)},
                {PTR_SZ, oskar_mem_buffer(out_v)}
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
