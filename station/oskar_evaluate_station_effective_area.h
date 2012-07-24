/*
 * Copyright (c) 2011, The University of Oxford
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


#ifndef OSKAR_EVALUATE_STATION_EFFECTIVE_AREA_H_
#define OSKAR_EVALUATE_STATION_EFFECTIVE_AREA_H_

/**
 * @file oskar_evaluate_station_effective_area.h
 */

#include "oskar_global.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates the approximate broadside effective area of an aperture
 * array station.
 *
 * @details
 * See: SKA Memo 87
 *
 * The effective area is evaluated according to the specified model.
 *
 * - OSKAR_EFFECTIVE_AREA_MODEL_SPARSE
 *      for arrays where: d > 2 * lambda
 *      A_eff ~ N * (lambda^2 / 2)
 *
 * - OSKAR_EFFECTIVE_AREA_MODEL_DENSE
 *      for arrays where: d < lambda/2
 *      A_eff ~ A_phy
 *
 *
 * @param[out] area          The station effective area, in m^2
 * @param[in] freq           Observation frequency, in Hz.
 * @param[in] num_antennas   Number of antenna elements in the AA.
 * @param[in] model          The effective area model to use. oskar_SettingsSky
 *                           enum: OSKAR_EFFECTIVE_AREA_MODEL_...
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_evaluate_station_effective_area(double* area, double freq,
        int num_antennas, int model);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_EFFECTIVE_AREA_H_ */
