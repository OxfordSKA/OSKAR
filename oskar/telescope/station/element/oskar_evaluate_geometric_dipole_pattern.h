/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#ifndef OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_
#define OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_

/**
 * @file oskar_evaluate_geometric_dipole_pattern.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions.
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of positions.
 * @param[in] theta              Point position (modified) theta values in rad.
 * @param[in] phi                Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output array (normally 1 or 4).
 * @param[in] offset             Start offset into output array.
 * @param[out] pattern           Array of output Jones matrices/scalars per source.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern(int num_points,
        const oskar_Mem* theta, const oskar_Mem* phi, int stride, int offset,
        oskar_Mem* pattern, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_H_ */
