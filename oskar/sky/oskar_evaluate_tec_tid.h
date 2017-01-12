/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_EVALUATE_TEC_TID_H_
#define OSKAR_EVALUATE_TEC_TID_H_

/**
 * @file oskar_evaluate_tec_tid.h
 */

#include <oskar_global.h>
#include <settings/old/oskar_Settings_old.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a TEC screen for a TID MIM.
 *
 * @param tec             Array of TEC values at the specified longitude and
 *                        latitude positions.
 * @param num_directions  Number of directions at which to evaluate the TEC.
 * @param lon             Array of longitudes, in radians.
 * @param lat             Array of latitudes, in radians.
 * @param rel_path_length Array of relative path lengths in the direction of
 *                        pierce point from the station.
 * @param TEC0            Zero offset TEC value.
 * @param TID             Settings structure describing the TID screen
 *                        components.
 * @param gast
 */
OSKAR_EXPORT
void oskar_evaluate_tec_tid(oskar_Mem* tec, int num_directions,
        const oskar_Mem* lon, const oskar_Mem* lat,
        const oskar_Mem* rel_path_length, double TEC0,
        oskar_SettingsTIDscreen* TID, double gast);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_TEC_TID_H_ */
