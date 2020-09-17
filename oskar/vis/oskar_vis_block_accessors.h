/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_VIS_BLOCK_ACCESSORS_H_
#define OSKAR_VIS_BLOCK_ACCESSORS_H_

/**
 * @file oskar_vis_block_accessors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
int oskar_vis_block_location(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_num_baselines(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_num_channels(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_num_stations(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_num_times(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_num_pols(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_start_channel_index(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_start_time_index(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_has_auto_correlations(const oskar_VisBlock* vis);

OSKAR_EXPORT
int oskar_vis_block_has_cross_correlations(const oskar_VisBlock* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_baseline_uu_metres(oskar_VisBlock* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_baseline_uu_metres_const(
        const oskar_VisBlock* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_baseline_vv_metres(oskar_VisBlock* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_baseline_vv_metres_const(
        const oskar_VisBlock* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_baseline_ww_metres(oskar_VisBlock* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_baseline_ww_metres_const(
        const oskar_VisBlock* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_baseline_uvw_metres(oskar_VisBlock* vis, int dim);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_baseline_uvw_metres_const(
        const oskar_VisBlock* vis, int dim);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_station_uvw_metres(oskar_VisBlock* vis, int dim);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_station_uvw_metres_const(
        const oskar_VisBlock* vis, int dim);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_auto_correlations(oskar_VisBlock* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_auto_correlations_const(
        const oskar_VisBlock* vis);

OSKAR_EXPORT
oskar_Mem* oskar_vis_block_cross_correlations(oskar_VisBlock* vis);

OSKAR_EXPORT
const oskar_Mem* oskar_vis_block_cross_correlations_const(
        const oskar_VisBlock* vis);

OSKAR_EXPORT
void oskar_vis_block_set_num_channels(oskar_VisBlock* vis,
        int value, int* status);

OSKAR_EXPORT
void oskar_vis_block_set_num_times(oskar_VisBlock* vis,
        int value, int* status);

OSKAR_EXPORT
void oskar_vis_block_set_start_channel_index(oskar_VisBlock* vis,
        int global_index);

OSKAR_EXPORT
void oskar_vis_block_set_start_time_index(oskar_VisBlock* vis,
        int global_index);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
