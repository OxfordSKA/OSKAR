/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "apps/lib/oskar_vis_block_write_ms.h"

#include <oskar_measurement_set.h>
#include <oskar_vis_block.h>
#include <oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_write_ms(const oskar_VisBlock* blk,
        const oskar_VisHeader* header, oskar_MeasurementSet* ms,
        int* status)
{
    const oskar_Mem *vis_amp, *uu, *vv, *ww;
    oskar_Mem *scratch;
    double dt_dump, t_start_sec;
    int num_baselines, num_pols_in, num_pols_out, num_channels, num_times;
    int b, c, i_in, i_out, offset, precision, row, start_row, t;
    const int *baseline_s1, *baseline_s2;
    const void* in;
    void* out;

    /* Check all inputs. */
    if (!blk || !header || !ms || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Pull data from visibility structure. */
    num_pols_out  = oskar_ms_num_pols(ms);
    num_pols_in   = oskar_vis_block_num_pols(blk);
    num_baselines = oskar_vis_block_num_baselines(blk);
    num_channels  = oskar_vis_block_num_channels(blk);
    num_times     = oskar_vis_block_num_times(blk);
    dt_dump       = oskar_vis_header_time_inc_sec(header);
    t_start_sec   = oskar_vis_block_time_start_mjd_utc_sec(blk) + 0.5 * dt_dump;
    vis_amp       = oskar_vis_block_amplitude_const(blk);
    uu            = oskar_vis_block_baseline_uu_metres_const(blk);
    vv            = oskar_vis_block_baseline_vv_metres_const(blk);
    ww            = oskar_vis_block_baseline_ww_metres_const(blk);
    baseline_s1   = oskar_vis_block_baseline_station1_const(blk);
    baseline_s2   = oskar_vis_block_baseline_station2_const(blk);
    precision     = oskar_mem_precision(vis_amp);

    /* Check polarisation dimension consistency:
     * num_pols_in can be less than num_pols_out, but not vice-versa. */
    if (num_pols_in > num_pols_out)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Set size of the main table. */
    start_row = oskar_ms_num_rows(ms);
    oskar_ms_set_num_rows(ms, start_row + num_times * num_baselines);

    /* Add visibilities and u,v,w coordinates. */
    scratch = oskar_mem_create(precision | OSKAR_COMPLEX, OSKAR_CPU,
            num_baselines * num_channels * num_pols_out, status);
    in  = oskar_mem_void_const(vis_amp);
    out = oskar_mem_void(scratch);
    if (precision == OSKAR_DOUBLE)
    {
        const double *uu_, *vv_, *ww_;
        uu_ = oskar_mem_double_const(uu, status);
        vv_ = oskar_mem_double_const(vv, status);
        ww_ = oskar_mem_double_const(ww, status);
        for (t = 0; t < num_times; ++t)
        {
            /* Construct the amplitude data for the given time. */
            row = t * num_baselines;
            offset = row * num_channels;
            if (num_pols_in == 4)
            {
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        ((double4c*)out)[i_out] = ((const double4c*)in)[i_in];
                    }
            }
            else if (num_pols_in == 1 && num_pols_out == 1)
            {
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        ((double2*)out)[i_out] = ((const double2*)in)[i_in];
                    }
            }
            else
            {
                double2 vis_amp, *out_;
                out_ = (double2*)out;
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        vis_amp = ((const double2*)in)[i_in];
                        out_[i_out + 0] = vis_amp;  /* XX */
                        out_[i_out + 1].x = 0.0;    /* XY */
                        out_[i_out + 1].y = 0.0;    /* XY */
                        out_[i_out + 2].x = 0.0;    /* YX */
                        out_[i_out + 2].y = 0.0;    /* YX */
                        out_[i_out + 3] = vis_amp;  /* YY */
                    }
            }
            oskar_ms_write_all_for_time_d(ms, row + start_row,
                    num_baselines, &uu_[row], &vv_[row], &ww_[row],
                    (const double*)out, baseline_s1, baseline_s2,
                    dt_dump, dt_dump, t_start_sec + (t * dt_dump));
        }
    }
    else if (precision == OSKAR_SINGLE)
    {
        const float *uu_, *vv_, *ww_;
        uu_ = oskar_mem_float_const(uu, status);
        vv_ = oskar_mem_float_const(vv, status);
        ww_ = oskar_mem_float_const(ww, status);
        for (t = 0; t < num_times; ++t)
        {
            /* Construct the amplitude data for the given time. */
            row = t * num_baselines;
            offset = row * num_channels;
            if (num_pols_in == 4)
            {
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        ((float4c*)out)[i_out] = ((const float4c*)in)[i_in];
                    }
            }
            else if (num_pols_in == 1 && num_pols_out == 1)
            {
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        ((float2*)out)[i_out] = ((const float2*)in)[i_in];
                    }
            }
            else
            {
                float2 vis_amp, *out_;
                out_ = (float2*)out;
                for (b = 0; b < num_baselines; ++b)
                    for (c = 0; c < num_channels; ++c)
                    {
                        i_in = offset + num_baselines * c + b;
                        i_out = num_channels * b + c;
                        vis_amp = ((const float2*)in)[i_in];
                        out_[i_out + 0] = vis_amp;  /* XX */
                        out_[i_out + 1].x = 0.0;    /* XY */
                        out_[i_out + 1].y = 0.0;    /* XY */
                        out_[i_out + 2].x = 0.0;    /* YX */
                        out_[i_out + 2].y = 0.0;    /* YX */
                        out_[i_out + 3] = vis_amp;  /* YY */
                    }
            }
            oskar_ms_write_all_for_time_f(ms, row + start_row,
                    num_baselines, &uu_[row], &vv_[row], &ww_[row],
                    (const float*)out, baseline_s1, baseline_s2,
                    dt_dump, dt_dump, t_start_sec + (t * dt_dump));
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* Cleanup. */
    oskar_mem_free(scratch, status);
}

#ifdef __cplusplus
}
#endif
