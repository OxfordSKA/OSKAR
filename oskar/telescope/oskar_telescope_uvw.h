/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_UVW_H_
#define OSKAR_TELESCOPE_UVW_H_

/**
 * @file oskar_telescope_uvw.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates (u,v,w) coordinates using observation parameters.
 *
 * @details
 * This function evaluates the station and, optionally, baseline (u,v,w)
 * coordinates using the supplied telescope model and simulation time
 * parameters.
 *
 * The baseline coordinates are not computed if \p uu, \p vv or \p ww are NULL.
 *
 * @param[in]  tel              Telescope model.
 * @param[in]  use_true coords  If set, use true station coordinates.
 * @param[in]  ignore_w_components If true, set all output w coordinates to 0.
 * @param[in]  num_times        Number of time steps to loop over.
 * @param[in]  time_ref_mjd_utc Start time of the observation.
 * @param[in]  time_inc_days    Time interval, in days.
 * @param[in]  start_time_index Time index for the start of the block.
 * @param[out] u                Output station u coordinates.
 * @param[out] v                Output station v coordinates.
 * @param[out] w                Output station w coordinates.
 * @param[out] uu               Optional output baseline u coordinates.
 * @param[out] vv               Optional output baseline v coordinates.
 * @param[out] ww               Optional output baseline w coordinates.
 * @param[in,out]  status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_uvw(
        const oskar_Telescope* tel,
        int use_true_coords,
        int ignore_w_components,
        int num_times,
        double time_ref_mjd_utc,
        double time_inc_days,
        int start_time_index,
        oskar_Mem* u,
        oskar_Mem* v,
        oskar_Mem* w,
        oskar_Mem* uu,
        oskar_Mem* vv,
        oskar_Mem* ww,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
