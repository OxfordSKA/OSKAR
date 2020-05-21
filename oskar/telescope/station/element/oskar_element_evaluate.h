/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#ifndef OSKAR_ELEMENT_EVALUATE_H_
#define OSKAR_ELEMENT_EVALUATE_H_

/**
 * @file oskar_element_evaluate.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the element model at the given source positions.
 *
 * @details
 * This function evaluates the element pattern model at the given source
 * positions.
 *
 * @param[in] model         Pointer to element model structure.
 * @param[in] normalise     If true, normalise pattern to value at zenith.
 * @param[in] swap_xy       If true, swap X and Y responses in output.
 * @param[in] orientation_x Azimuth of X dipole in radians.
 * @param[in] orientation_y Azimuth of Y dipole in radians.
 * @param[in] offset_points Start offset into input coordinate arrays.
 * @param[in] num_points    Number of points at which to evaluate beam.
 * @param[in] x             Pointer to x-direction cosines.
 * @param[in] y             Pointer to y-direction cosines.
 * @param[in] z             Pointer to z-direction cosines.
 * @param[in] frequency_hz  Current observing frequency in Hz.
 * @param[in,out] theta     Pointer to work array for computing theta values.
 * @param[in,out] phi_x     Pointer to work array for computing phi values.
 * @param[in,out] phi_y     Pointer to work array for computing phi values.
 * @param[in] offset_out    Start offset into output array.
 * @param[in,out] output    Pointer to output array.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_element_evaluate(
        const oskar_Element* model,
        int normalise,
        int swap_xy,
        double orientation_x,
        double orientation_y,
        int offset_points,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        double frequency_hz,
        oskar_Mem* theta,
        oskar_Mem* phi_x,
        oskar_Mem* phi_y,
        int offset_out,
        oskar_Mem* output,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
