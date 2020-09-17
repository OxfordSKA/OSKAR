/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

#define D2R (M_PI / 180.0)

/* Local helper macros. */

#define ASSEMBLE_ALL_FOR_TIME(FP, FP2, FP4c) {\
    unsigned int a1, a2, b, c, j;\
    if (start_chan_index == 0) {\
        /* Assemble baseline coordinates. */\
        for (a1 = 0, b = 0, j = 0; a1 < num_stations; ++a1) {\
            if (have_auto) {\
                ((FP*)uu_out)[j] = ((FP*)vv_out)[j] = ((FP*)ww_out)[j] = 0.0;\
                ++j;\
            }\
            if (have_cross) {\
                for (a2 = a1 + 1; a2 < num_stations; ++a2, ++b, ++j) {\
                    const unsigned int i = num_baseln_in * t + b;\
                    ((FP*)uu_out)[j] = ((const FP*)uu_in)[i];\
                    ((FP*)vv_out)[j] = ((const FP*)vv_in)[i];\
                    ((FP*)ww_out)[j] = ((const FP*)ww_in)[i];\
                }\
            }\
        }\
    }\
    /* Assemble visibilities. */\
    for (c = 0, j = 0; c < num_channels; ++c) {\
        const unsigned int ia = num_stations * (t * num_channels + c);\
        const unsigned int ix = num_baseln_in * (t * num_channels + c);\
        if (num_pols_in == 4) ASSEMBLE_VIS(FP4c)\
        else if (num_pols_out == 1) ASSEMBLE_VIS(FP2)\
        else {\
            FP2 zero; zero.x = zero.y = 0.0;\
            for (a1 = 0, b = 0; a1 < num_stations; ++a1) {\
                if (have_auto) COPY_SCALAR(FP2, acorr, ia + a1)\
                if (have_cross)\
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)\
                        COPY_SCALAR(FP2, xcorr, ix + b)\
            }\
        }\
    }\
}

#define ASSEMBLE_VIS(T) {\
            for (a1 = 0, b = 0; a1 < num_stations; ++a1) {\
                if (have_auto) ((T*)out)[j++] = ((const T*)acorr)[ia + a1];\
                if (have_cross)\
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)\
                        ((T*)out)[j++] = ((const T*)xcorr)[ix + b];\
            }\
        }\

#define COPY_SCALAR(FP2, DATA, IDX) {\
                    const FP2 val = ((const FP2*)DATA)[IDX];\
                    ((FP2*)out)[j + 0] = val;  ((FP2*)out)[j + 1] = zero;\
                    ((FP2*)out)[j + 2] = zero; ((FP2*)out)[j + 3] = val;\
                    j += 4;\
                }\


void oskar_vis_block_write_ms(const oskar_VisBlock* blk,
        const oskar_VisHeader* header, oskar_MeasurementSet* ms, int* status)
{
    const oskar_Mem *in_acorr, *in_xcorr, *in_uu, *in_vv, *in_ww;
    oskar_Mem *temp_vis = 0, *temp_uu = 0, *temp_vv = 0, *temp_ww = 0;
    double exposure_sec, interval_sec, t_start_mjd, t_start_sec;
    double ra_rad, dec_rad, freq_start_hz;
    unsigned int num_baseln_in, num_baseln_out, num_channels;
    unsigned int num_pols_in, num_pols_out, num_stations, num_times, t;
    unsigned int prec, start_time_index, start_chan_index;
    unsigned int have_auto, have_cross;
    const void *uu_in, *vv_in, *ww_in, *xcorr, *acorr;
    void *uu_out, *vv_out, *ww_out, *out;
    if (*status) return;

    /* Pull data from visibility structures. */
    num_pols_out     = oskar_ms_num_pols(ms);
    num_pols_in      = oskar_vis_block_num_pols(blk);
    num_stations     = oskar_vis_block_num_stations(blk);
    num_baseln_in    = oskar_vis_block_num_baselines(blk);
    num_channels     = oskar_vis_block_num_channels(blk);
    num_times        = oskar_vis_block_num_times(blk);
    in_acorr         = oskar_vis_block_auto_correlations_const(blk);
    in_xcorr         = oskar_vis_block_cross_correlations_const(blk);
    in_uu            = oskar_vis_block_baseline_uu_metres_const(blk);
    in_vv            = oskar_vis_block_baseline_vv_metres_const(blk);
    in_ww            = oskar_vis_block_baseline_ww_metres_const(blk);
    have_auto        = oskar_vis_block_has_auto_correlations(blk);
    have_cross       = oskar_vis_block_has_cross_correlations(blk);
    start_time_index = oskar_vis_block_start_time_index(blk);
    start_chan_index = oskar_vis_block_start_channel_index(blk);
    ra_rad           = oskar_vis_header_phase_centre_ra_deg(header) * D2R;
    dec_rad          = oskar_vis_header_phase_centre_dec_deg(header) * D2R;
    exposure_sec     = oskar_vis_header_time_average_sec(header);
    interval_sec     = oskar_vis_header_time_inc_sec(header);
    t_start_mjd      = oskar_vis_header_time_start_mjd_utc(header);
    freq_start_hz    = oskar_vis_header_freq_start_hz(header);
    prec             = oskar_mem_precision(in_xcorr);
    t_start_sec      = t_start_mjd * 86400.0;

    /* Check that there is something to write. */
    if (!have_auto && !have_cross) return;

    /* Get number of output baselines. */
    num_baseln_out = num_baseln_in;
    if (have_auto)
        num_baseln_out += num_stations;

    /* Check polarisation dimension consistency:
     * num_pols_in can be less than num_pols_out, but not vice-versa. */
    if (num_pols_in > num_pols_out)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check the dimensions match. */
    if (oskar_ms_num_pols(ms) != num_pols_out ||
            oskar_ms_num_stations(ms) != num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check the reference frequencies match. */
    if (fabs(oskar_ms_freq_start_hz(ms) - freq_start_hz) > 1e-10)
    {
        *status = OSKAR_ERR_VALUE_MISMATCH;
        return;
    }

    /* Check the phase centres are the same. */
    if (fabs(oskar_ms_phase_centre_ra_rad(ms) - ra_rad) > 1e-10 ||
            fabs(oskar_ms_phase_centre_dec_rad(ms) - dec_rad) > 1e-10)
    {
        *status = OSKAR_ERR_VALUE_MISMATCH;
        return;
    }

    /* Write visibilities and (u,v,w) coordinates. */
    temp_vis = oskar_mem_create(prec | OSKAR_COMPLEX, OSKAR_CPU,
            num_baseln_out * num_channels * num_pols_out, status);
    temp_uu = oskar_mem_create(prec, OSKAR_CPU, num_baseln_out, status);
    temp_vv = oskar_mem_create(prec, OSKAR_CPU, num_baseln_out, status);
    temp_ww = oskar_mem_create(prec, OSKAR_CPU, num_baseln_out, status);
    out     = oskar_mem_void(temp_vis);
    uu_out  = oskar_mem_void(temp_uu);
    vv_out  = oskar_mem_void(temp_vv);
    ww_out  = oskar_mem_void(temp_ww);
    xcorr   = oskar_mem_void_const(in_xcorr);
    acorr   = oskar_mem_void_const(in_acorr);
    uu_in   = oskar_mem_void_const(in_uu);
    vv_in   = oskar_mem_void_const(in_vv);
    ww_in   = oskar_mem_void_const(in_ww);
    if (prec == OSKAR_DOUBLE)
    {
        for (t = 0; t < num_times; ++t)
        {
            /* Assemble the baseline coordinates and all visibilities
             * for the given time. */
            ASSEMBLE_ALL_FOR_TIME(double, double2, double4c)
            const unsigned int row0 = (start_time_index + t) * num_baseln_out;
            oskar_ms_write_vis_d(ms, row0, start_chan_index,
                    num_channels, num_baseln_out, (double*)out);

            /* Only write the coordinates for the first channel. */
            if (start_chan_index == 0)
                oskar_ms_write_coords_d(ms, row0, num_baseln_out,
                        (double*)uu_out, (double*)vv_out, (double*)ww_out,
                        exposure_sec, interval_sec,
                        (start_time_index + t + 0.5) * interval_sec +
                        t_start_sec);
        }
    }
    else if (prec == OSKAR_SINGLE)
    {
        for (t = 0; t < num_times; ++t)
        {
            /* Assemble the baseline coordinates and all visibilities
             * for the given time. */
            ASSEMBLE_ALL_FOR_TIME(float, float2, float4c)
            const unsigned int row0 = (start_time_index + t) * num_baseln_out;
            oskar_ms_write_vis_f(ms, row0, start_chan_index,
                    num_channels, num_baseln_out, (float*)out);

            /* Only write the coordinates for the first channel. */
            if (start_chan_index == 0)
                oskar_ms_write_coords_f(ms, row0, num_baseln_out,
                        (float*)uu_out, (float*)vv_out, (float*)ww_out,
                        exposure_sec, interval_sec,
                        (start_time_index + t + 0.5) * interval_sec +
                        t_start_sec);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* Cleanup. */
    oskar_mem_free(temp_vis, status);
    oskar_mem_free(temp_uu, status);
    oskar_mem_free(temp_vv, status);
    oskar_mem_free(temp_ww, status);
}

#ifdef __cplusplus
}
#endif
