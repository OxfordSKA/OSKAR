/*
 * Copyright (c) 2013-2019, The University of Oxford
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
void oskar_convert_station_uvw_to_baseline_uvw(int num_stations, int offset_in,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        int offset_out, oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW_H_ */
