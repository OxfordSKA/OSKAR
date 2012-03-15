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


#ifndef OSKAR_PROJECT_GAUSSIAN_SPH_TO_LM_H_
#define OSKAR_PROJECT_GAUSSIAN_SPH_TO_LM_H_

/**
 * @file oskar_project_gaussian_sph_to_lm.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluate the parameters of a gaussian projected onto the tangent plane
 * given a description of a gaussian defined on the surface of a sphere.
 *
 * Note: All values are in radians.
 *
 * @param[out]  maj       Major axis of gaussian on tangent plane.
 * @param[out]  min       Minor axis of gaussian on tangent plane.
 * @param[out]  pa        Position angle of gaussian on tangent plane.
 * @param[in]   sph_maj   Major axis of gaussian on sphere.
 * @param[in]   sph_min   Minor axis of gaussian on sphere.
 * @param[in]   sph_pa    Position angle of gaussian on sphere.
 * @param[in]   lon       Position of gaussian on the sphere.
 * @param[in]   lat       Position of gaussian on the sphere.
 * @param[in]   lon0      Origin of tangent plane
 * @param[in]   lat0      Origin of tangent plane
 *
 * @return An error code
 */
OSKAR_EXPORT
int oskar_project_gaussian_sph_to_lm(int num_points, oskar_Mem* maj,
        oskar_Mem* min, oskar_Mem* pa, const oskar_Mem* sph_maj,
        const oskar_Mem* sph_min, const oskar_Mem* sph_pa, const oskar_Mem* lon,
        const oskar_Mem* lat, double lon0, double lat0);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PROJECT_GAUSSIAN_SPH_TO_LM_H_ */
