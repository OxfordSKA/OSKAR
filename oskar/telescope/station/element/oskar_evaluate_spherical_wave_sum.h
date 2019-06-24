/*
 * Copyright (c) 2019, The University of Oxford
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

#ifndef OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_H_
#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_H_

/**
 * @file oskar_evaluate_spherical_wave_sum.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the sum of spherical wave coefficients.
 *
 * @details
 * Evaluate the sum of spherical wave coefficients at the
 * given coordinates.
 *
 * @param[in] num_points    Number of coordinate points.
 * @param[in] theta         Coordinate theta (polar) values, in radians.
 * @param[in] phi_x         Coordinate phi (azimuthal) values for X, in radians.
 * @param[in] phi_y         Coordinate phi (azimuthal) values for Y, in radians.
 * @param[in] l_max         Maximum order of spherical wave.
 * @param[in] alpha         TE and TM mode coefficients for X and Y antennas.
 * @param[in] offset        Offset into output data array.
 * @param[in,out] pattern   Output data array of length at least \p num_points.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_spherical_wave_sum(int num_points, const oskar_Mem* theta,
        const oskar_Mem* phi_x, const oskar_Mem* phi_y, int l_max,
        const oskar_Mem* alpha, int offset, oskar_Mem* pattern, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
