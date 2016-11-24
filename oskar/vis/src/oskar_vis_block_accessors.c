/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif


int oskar_vis_block_location(const oskar_VisBlock* vis)
{
    return oskar_mem_location(vis->cross_correlations);
}

int oskar_vis_block_num_baselines(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[4];
}

int oskar_vis_block_num_channels(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[3];
}

int oskar_vis_block_num_stations(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[5];
}

int oskar_vis_block_num_times(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[2];
}

int oskar_vis_block_num_pols(const oskar_VisBlock* vis)
{
    return oskar_mem_is_matrix(vis->cross_correlations) ? 4 : 1;
}

int oskar_vis_block_start_channel_index(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[1];
}

int oskar_vis_block_start_time_index(const oskar_VisBlock* vis)
{
    return vis->dim_start_size[0];
}

int oskar_vis_block_has_auto_correlations(const oskar_VisBlock* vis)
{
    return (oskar_mem_length(vis->auto_correlations) > 0);
}

int oskar_vis_block_has_cross_correlations(const oskar_VisBlock* vis)
{
    return (oskar_mem_length(vis->cross_correlations) > 0);
}

oskar_Mem* oskar_vis_block_baseline_uu_metres(oskar_VisBlock* vis)
{
    return vis->baseline_uu_metres;
}

const oskar_Mem* oskar_vis_block_baseline_uu_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_uu_metres;
}

oskar_Mem* oskar_vis_block_baseline_vv_metres(oskar_VisBlock* vis)
{
    return vis->baseline_vv_metres;
}

const oskar_Mem* oskar_vis_block_baseline_vv_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_vv_metres;
}

oskar_Mem* oskar_vis_block_baseline_ww_metres(oskar_VisBlock* vis)
{
    return vis->baseline_ww_metres;
}

const oskar_Mem* oskar_vis_block_baseline_ww_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_ww_metres;
}

oskar_Mem* oskar_vis_block_auto_correlations(oskar_VisBlock* vis)
{
    return vis->auto_correlations;
}

const oskar_Mem* oskar_vis_block_auto_correlations_const(
        const oskar_VisBlock* vis)
{
    return vis->auto_correlations;
}

oskar_Mem* oskar_vis_block_cross_correlations(oskar_VisBlock* vis)
{
    return vis->cross_correlations;
}

const oskar_Mem* oskar_vis_block_cross_correlations_const(
        const oskar_VisBlock* vis)
{
    return vis->cross_correlations;
}

void oskar_vis_block_set_num_channels(oskar_VisBlock* vis,
        int value, int* status)
{
    oskar_vis_block_resize(vis, oskar_vis_block_num_times(vis),
            value, oskar_vis_block_num_stations(vis), status);
}

void oskar_vis_block_set_num_times(oskar_VisBlock* vis,
        int value, int* status)
{
    oskar_vis_block_resize(vis, value,
            oskar_vis_block_num_channels(vis),
            oskar_vis_block_num_stations(vis), status);
}

void oskar_vis_block_set_start_channel_index(oskar_VisBlock* vis,
        int global_index)
{
    vis->dim_start_size[1] = global_index;
}

void oskar_vis_block_set_start_time_index(oskar_VisBlock* vis,
        int global_index)
{
    vis->dim_start_size[0] = global_index;
}

#ifdef __cplusplus
}
#endif
