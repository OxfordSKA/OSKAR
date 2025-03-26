/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW_H_
#define OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW_H_

/**
 * @file oskar_convert_station_uvw_to_baseline_uvw.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the baseline (u,v,w) coordinates for all station pairs.
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this function computes
 * the baseline coordinates for all station pairs.
 *
 * @param[in]  use_casa_phase_convention If set, use the CASA phase convention.
 * @param[in]  num_stations  Number of stations.
 * @param[in]  offset_in     Input array offset.
 * @param[in]  u             Station u coordinates.
 * @param[in]  v             Station v coordinates.
 * @param[in]  w             Station w coordinates.
 * @param[in]  offset_out    Output array offset.
 * @param[out] uu            Baseline u coordinates.
 * @param[out] vv            Baseline v coordinates.
 * @param[out] ww            Baseline w coordinates.
 */
OSKAR_EXPORT
void oskar_convert_station_uvw_to_baseline_uvw(
        int use_casa_phase_convention,
        int num_stations,
        int offset_in,
        const oskar_Mem* u,
        const oskar_Mem* v,
        const oskar_Mem* w,
        int offset_out,
        oskar_Mem* uu,
        oskar_Mem* vv,
        oskar_Mem* ww,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW_H_ */
