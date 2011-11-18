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

#ifndef OSKAR_SPH_FROM_LM_H_
#define OSKAR_SPH_FROM_LM_H_

/**
 * @file oskar_sph_from_lm.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert direction cosines to angles (single precision).
 *
 * @details
 * Returns the longitude and latitude of the supplied array of
 * direction cosines (l and m).
 *
 * @param[in]  num_positions Number of positions to evaluate.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  l             Array of positions in cosine space.
 * @param[in]  m             Array of positions in cosine space.
 * @param[out] lon           Array of longitude values, in radians.
 * @param[out] lat           Array of latitude values, in radians.
 */
OSKAR_EXPORT
void oskar_sph_from_lm_f(int num_positions, float lon0, float lat0,
        const float* l, const float* m, float* lon, float* lat);

/**
 * @brief
 * Convert direction cosines to angles (double precision).
 *
 * @details
 * Returns the longitude and latitude of the supplied array of
 * direction cosines (l and m).
 *
 * @param[in]  num_positions Number of positions to evaluate.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  l             Array of positions in cosine space.
 * @param[in]  m             Array of positions in cosine space.
 * @param[out] lon           Array of longitude values, in radians.
 * @param[out] lat           Array of latitude values, in radians.
 */
OSKAR_EXPORT
void oskar_sph_from_lm_d(int num_positions, double lon0, double lat0,
        const double* l, const double* m, double* lon, double* lat);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPH_FROM_LM_H_ */
