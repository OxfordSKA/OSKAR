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

#include <private_vis_block.h>
#include <oskar_vis_block.h>

#ifdef __cplusplus
extern "C" {
#endif


int oskar_vis_block_location(const oskar_VisBlock* vis)
{
    return oskar_mem_location(vis->amplitude);
}

int oskar_vis_block_num_baselines(const oskar_VisBlock* vis)
{
    return vis->dim_size[2];
}

int oskar_vis_block_num_channels(const oskar_VisBlock* vis)
{
    return vis->dim_size[1];
}

int oskar_vis_block_num_stations(const oskar_VisBlock* vis)
{
    return vis->dim_size[3];
}

int oskar_vis_block_num_times(const oskar_VisBlock* vis)
{
    return vis->dim_size[0];
}

void oskar_vis_block_set_num_times(oskar_VisBlock* vis, int value, int* status)
{
    if (!status || *status != OSKAR_SUCCESS)
        return;

    /* Only allow shrinking of the dimension in order to avoid issues
     * of the risk of having to resize the memory given the block capacity is
     * not known.
     */
    if (value <= vis->dim_size[0]) {
        vis->dim_size[0] = value;
    }
    else {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }
}

int oskar_vis_block_num_pols(const oskar_VisBlock* vis)
{
    return oskar_mem_is_matrix(vis->amplitude) ? 4 : 1;
}

double oskar_vis_block_freq_start_hz(const oskar_VisBlock* vis)
{
    return vis->freq_range_hz[0];
}

double oskar_vis_block_freq_end_hz(const oskar_VisBlock* vis)
{
    return vis->freq_range_hz[1];
}

double oskar_vis_block_time_start_mjd_utc_sec(const oskar_VisBlock* vis)
{
    return vis->time_range_mjd_utc_sec[0];
}

double oskar_vis_block_time_end_mjd_utc_sec(const oskar_VisBlock* vis)
{
    return vis->time_range_mjd_utc_sec[1];
}

oskar_Mem* oskar_vis_block_baseline_uu_metres(oskar_VisBlock* vis)
{
    return vis->baseline_uu_metres;
}

const oskar_Mem* oskar_vis_block_baseline_uu_metres_const(const oskar_VisBlock* vis)
{
    return vis->baseline_uu_metres;
}

oskar_Mem* oskar_vis_block_baseline_vv_metres(oskar_VisBlock* vis)
{
    return vis->baseline_vv_metres;
}

const oskar_Mem* oskar_vis_block_baseline_vv_metres_const(const oskar_VisBlock* vis)
{
    return vis->baseline_vv_metres;
}

oskar_Mem* oskar_vis_block_baseline_ww_metres(oskar_VisBlock* vis)
{
    return vis->baseline_ww_metres;
}

const oskar_Mem* oskar_vis_block_baseline_ww_metres_const(const oskar_VisBlock* vis)
{
    return vis->baseline_ww_metres;
}

oskar_Mem* oskar_vis_block_amplitude(oskar_VisBlock* vis)
{
    return vis->amplitude;
}

const oskar_Mem* oskar_vis_block_amplitude_const(const oskar_VisBlock* vis)
{
    return vis->amplitude;
}

const int* oskar_vis_block_baseline_station1_const(const oskar_VisBlock* vis)
{
    int status = 0;
    return oskar_mem_int_const(vis->a1, &status);
}

const int* oskar_vis_block_baseline_station2_const(const oskar_VisBlock* vis)
{
    int status = 0;
    return oskar_mem_int_const(vis->a2, &status);
}

void oskar_vis_block_set_freq_range_hz(oskar_VisBlock* vis,
        double start, double end)
{
    vis->freq_range_hz[0] = start;
    vis->freq_range_hz[1] = end;
}

void oskar_vis_block_set_time_range_mjd_utc_sec(oskar_VisBlock* vis,
        double start, double end)
{
    vis->time_range_mjd_utc_sec[0] = start;
    vis->time_range_mjd_utc_sec[1] = end;
}

#ifdef __cplusplus
}
#endif
