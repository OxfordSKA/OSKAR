/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_SKY_EVALUATE_RELATIVE_DIRECTIONS_H_
#define OSKAR_SKY_EVALUATE_RELATIVE_DIRECTIONS_H_

/**
 * @file oskar_sky_evaluate_relative_directions.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates 3D direction cosines of sources relative to phase centre.
 *
 * @details
 * This function populates the 3D direction cosines (l,m,n coordinates)
 * of all sources relative to the phase centre.
 *
 * It assumes that the source RA and Dec positions have already been filled,
 * and that the arrays have been preallocated to the correct length.
 *
 * @param[in,out] sky    Pointer to sky model structure.
 * @param[in] ra0_rad    Right Ascension of phase centre, in radians.
 * @param[in] dec0_rad   Declination of phase centre, in radians.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_sky_evaluate_relative_directions(oskar_Sky* sky, double ra0_rad,
        double dec0_rad, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_EVALUATE_RELATIVE_DIRECTIONS_H_ */
