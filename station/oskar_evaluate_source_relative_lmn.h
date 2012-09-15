/*
 * Copyright (c) 2012, The University of Oxford
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


#ifndef OSKAR_EVALUATE_SOURCE_RELATIVE_LMN_H_
#define OSKAR_EVALUATE_SOURCE_RELATIVE_LMN_H_

/**
 * @file oskar_evaluate_source_relative_lmn.h
 */

#include "oskar_global.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates relative direction cosines for specified sources.
 *
 * @details
 * This function converts source positions from equatorial (RA, Dec)
 * coordinates to direction cosines relative to the beam pointing direction
 * specified in the station model.
 *
 * @param[in] num_sources The number of source positions to process.
 * @param[out] l          Source relative l direction cosines (x-components).
 * @param[out] m          Source relative m direction cosines (y-components).
 * @param[out] n          Source relative n direction cosines (z-components).
 * @param[in]  RA         Source Right Ascension values.
 * @param[in]  Dec        Source Declination values.
 * @param[in]  station    Pointer to station model.
 * @param[in,out]  status Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_source_relative_lmn(int num_sources, oskar_Mem* l,
        oskar_Mem* m, oskar_Mem* n, const oskar_Mem* RA, const oskar_Mem* Dec,
        const oskar_StationModel* station, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_SOURCE_RELATIVE_LMN_H_ */
