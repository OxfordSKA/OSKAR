/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
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
    const void* nullp = 0;
    const int type = oskar_sky_int(sky, OSKAR_SKY_PRECISION);
    const int location = oskar_sky_int(sky, OSKAR_SKY_MEM_LOCATION);
    const int num_sources = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    const int num_spx_values = oskar_sky_num_columns_of_type(
            sky, OSKAR_SKY_SPEC_IDX
    );
    const oskar_Mem* in_i = oskar_sky_column_const(sky, OSKAR_SKY_I_JY, 0);
    const oskar_Mem* in_q = oskar_sky_column_const(sky, OSKAR_SKY_Q_JY, 0);
    const oskar_Mem* in_u = oskar_sky_column_const(sky, OSKAR_SKY_U_JY, 0);
    const oskar_Mem* in_v = oskar_sky_column_const(sky, OSKAR_SKY_V_JY, 0);
    const oskar_Mem* ref_hz = oskar_sky_column_const(sky, OSKAR_SKY_REF_HZ, 0);
    const oskar_Mem* lin_si = oskar_sky_column_const(sky, OSKAR_SKY_LIN_SI, 0);
    const oskar_Mem* spx0 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 0);
    const oskar_Mem* spx1 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 1);
    const oskar_Mem* spx2 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 2);
    const oskar_Mem* spx3 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 3);
    const oskar_Mem* spx4 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 4);
    const oskar_Mem* spx5 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 5);
    const oskar_Mem* spx6 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 6);
    const oskar_Mem* spx7 = oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 7);
    const oskar_Mem* rm = oskar_sky_column_const(sky, OSKAR_SKY_RM_RAD, 0);
    const oskar_Mem* pola = oskar_sky_column_const(sky, OSKAR_SKY_POLA_RAD, 0);
    const oskar_Mem* polf = oskar_sky_column_const(sky, OSKAR_SKY_POLF, 0);
    const oskar_Mem* refw = oskar_sky_column_const(
            sky, OSKAR_SKY_REF_WAVE_M, 0
    );
    const oskar_Mem* sp_curv = oskar_sky_column_const(
            sky, OSKAR_SKY_SPEC_CURV, 0
    );
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
    if (!in_i)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        oskar_log_error(0, "No Stokes I column in sky model.");
        return;
    }
    if (num_spx_values > 8)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        oskar_log_error(
                0, "A maximum of 8 spectral index terms are supported."
        );
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            scale_flux_with_frequency_float(
                    num_sources, freq_hz,
                    oskar_mem_float_const(in_i, status),
                    in_q ? oskar_mem_float_const(in_q, status) : nullp,
                    in_u ? oskar_mem_float_const(in_u, status) : nullp,
                    in_v ? oskar_mem_float_const(in_v, status) : nullp,
                    ref_hz ? oskar_mem_float_const(ref_hz, status) : nullp,
                    lin_si ? oskar_mem_float_const(lin_si, status) : nullp,
                    num_spx_values,
                    spx0 ? oskar_mem_float_const(spx0, status) : nullp,
                    spx1 ? oskar_mem_float_const(spx1, status) : nullp,
                    spx2 ? oskar_mem_float_const(spx2, status) : nullp,
                    spx3 ? oskar_mem_float_const(spx3, status) : nullp,
                    spx4 ? oskar_mem_float_const(spx4, status) : nullp,
                    spx5 ? oskar_mem_float_const(spx5, status) : nullp,
                    spx6 ? oskar_mem_float_const(spx6, status) : nullp,
                    spx7 ? oskar_mem_float_const(spx7, status) : nullp,
                    rm ? oskar_mem_float_const(rm, status) : nullp,
                    polf ? oskar_mem_float_const(polf, status) : nullp,
                    pola ? oskar_mem_float_const(pola, status) : nullp,
                    refw ? oskar_mem_float_const(refw, status) : nullp,
                    sp_curv ? oskar_mem_float_const(sp_curv, status) : nullp,
                    oskar_mem_float(out_i, status),
                    oskar_mem_float(out_q, status),
                    oskar_mem_float(out_u, status),
                    oskar_mem_float(out_v, status)
            );
        }
        else if (type == OSKAR_DOUBLE)
        {
            scale_flux_with_frequency_double(
                    num_sources, freq_hz,
                    oskar_mem_double_const(in_i, status),
                    in_q ? oskar_mem_double_const(in_q, status) : nullp,
                    in_u ? oskar_mem_double_const(in_u, status) : nullp,
                    in_v ? oskar_mem_double_const(in_v, status) : nullp,
                    ref_hz ? oskar_mem_double_const(ref_hz, status) : nullp,
                    lin_si ? oskar_mem_double_const(lin_si, status) : nullp,
                    num_spx_values,
                    spx0 ? oskar_mem_double_const(spx0, status) : nullp,
                    spx1 ? oskar_mem_double_const(spx1, status) : nullp,
                    spx2 ? oskar_mem_double_const(spx2, status) : nullp,
                    spx3 ? oskar_mem_double_const(spx3, status) : nullp,
                    spx4 ? oskar_mem_double_const(spx4, status) : nullp,
                    spx5 ? oskar_mem_double_const(spx5, status) : nullp,
                    spx6 ? oskar_mem_double_const(spx6, status) : nullp,
                    spx7 ? oskar_mem_double_const(spx7, status) : nullp,
                    rm ? oskar_mem_double_const(rm, status) : nullp,
                    polf ? oskar_mem_double_const(polf, status) : nullp,
                    pola ? oskar_mem_double_const(pola, status) : nullp,
                    refw ? oskar_mem_double_const(refw, status) : nullp,
                    sp_curv ? oskar_mem_double_const(sp_curv, status) : nullp,
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
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&freq_hz : (const void*)&freq_hz_f},
                {PTR_SZ, oskar_mem_buffer_const(in_i)},
                {PTR_SZ, in_q ? oskar_mem_buffer_const(in_q) : &nullp},
                {PTR_SZ, in_u ? oskar_mem_buffer_const(in_u) : &nullp},
                {PTR_SZ, in_v ? oskar_mem_buffer_const(in_v) : &nullp},
                {PTR_SZ, ref_hz ? oskar_mem_buffer_const(ref_hz) : &nullp},
                {PTR_SZ, lin_si ? oskar_mem_buffer_const(lin_si) : &nullp},
                {INT_SZ, &num_spx_values},
                {PTR_SZ, spx0 ? oskar_mem_buffer_const(spx0) : &nullp},
                {PTR_SZ, spx1 ? oskar_mem_buffer_const(spx1) : &nullp},
                {PTR_SZ, spx2 ? oskar_mem_buffer_const(spx2) : &nullp},
                {PTR_SZ, spx3 ? oskar_mem_buffer_const(spx3) : &nullp},
                {PTR_SZ, spx4 ? oskar_mem_buffer_const(spx4) : &nullp},
                {PTR_SZ, spx5 ? oskar_mem_buffer_const(spx5) : &nullp},
                {PTR_SZ, spx6 ? oskar_mem_buffer_const(spx6) : &nullp},
                {PTR_SZ, spx7 ? oskar_mem_buffer_const(spx7) : &nullp},
                {PTR_SZ, rm ? oskar_mem_buffer_const(rm) : &nullp},
                {PTR_SZ, polf ? oskar_mem_buffer_const(polf) : &nullp},
                {PTR_SZ, pola ? oskar_mem_buffer_const(pola) : &nullp},
                {PTR_SZ, refw ? oskar_mem_buffer_const(refw) : &nullp},
                {PTR_SZ, sp_curv ? oskar_mem_buffer_const(sp_curv) : &nullp},
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
