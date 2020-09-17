/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    return oskar_vis_block_baseline_uvw_metres(vis, 0);
}

const oskar_Mem* oskar_vis_block_baseline_uu_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_uvw_metres[0];
}

oskar_Mem* oskar_vis_block_baseline_vv_metres(oskar_VisBlock* vis)
{
    return oskar_vis_block_baseline_uvw_metres(vis, 1);
}

const oskar_Mem* oskar_vis_block_baseline_vv_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_uvw_metres[1];
}

oskar_Mem* oskar_vis_block_baseline_ww_metres(oskar_VisBlock* vis)
{
    return oskar_vis_block_baseline_uvw_metres(vis, 2);
}

const oskar_Mem* oskar_vis_block_baseline_ww_metres_const(
        const oskar_VisBlock* vis)
{
    return vis->baseline_uvw_metres[2];
}

oskar_Mem* oskar_vis_block_baseline_uvw_metres(oskar_VisBlock* vis, int dim)
{
    int status = 0;
    oskar_Mem* coords = vis->baseline_uvw_metres[dim];
    oskar_mem_ensure(coords, oskar_vis_block_num_times(vis) *
            oskar_vis_block_num_baselines(vis), &status);
    return coords;
}

const oskar_Mem* oskar_vis_block_baseline_uvw_metres_const(
        const oskar_VisBlock* vis, int dim)
{
    return vis->baseline_uvw_metres[dim];
}

oskar_Mem* oskar_vis_block_station_uvw_metres(oskar_VisBlock* vis, int dim)
{
    int status = 0;
    oskar_Mem* coords = vis->station_uvw_metres[dim];
    oskar_mem_ensure(coords, oskar_vis_block_num_times(vis) *
            oskar_vis_block_num_stations(vis), &status);
    return coords;
}

const oskar_Mem* oskar_vis_block_station_uvw_metres_const(
        const oskar_VisBlock* vis, int dim)
{
    return vis->station_uvw_metres[dim];
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
