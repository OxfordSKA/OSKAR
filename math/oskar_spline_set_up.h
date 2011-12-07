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

#ifndef OSKAR_SPLINE_SET_UP_H_
#define OSKAR_SPLINE_SET_UP_H_

/**
 * @file oskar_spline_set_up.h
 */

#include "oskar_global.h"
#include "math/oskar_SplineData.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets up spline knot positions and coefficients.
 *
 * @details
 * This function sets up knot positions and coefficients for a spline surface.
 *
 * It calls the FORTRAN function regrid from the DIERCKX library to perform the
 * maths.
 *
 * References:
 *  Dierckx P. : A fast algorithm for smoothing data on a rectangular
 *               grid while using spline functions, Siam J.Numer.Anal.
 *               19 (1982) 1286-1304.
 *  Dierckx P. : A fast algorithm for smoothing data on a rectangular
 *               grid while using spline functions, Report TW53, Dept.
 *               Computer Science, K.U.Leuven, 1980.
 *  Dierckx P. : Curve and surface fitting with splines, monographs on
 *               numerical analysis, Oxford University Press, 1993.
 *
 * Note that the fastest varying dimension is along y.
 *
 * The arrays \p tx and \p ty must be large enough to hold all the knot
 * positions, so set them to be (size_x + kx + 1) and (size_y + ky + 1),
 * respectively.
 *
 * The array \p c must be large enough to hold all the spline coefficients
 * at the knot positions, so set to it be (num_x * num_y).
 *
 * @param[in] num_x The number of input grid points in the x dimension.
 * @param[in] x     The input grid positions in x.
 * @param[in] num_y The number of input grid points in the y dimension.
 * @param[in] y     The input grid positions in y.
 * @param[in] data  The data table (dimensions num_x * num_y).
 * @param[in] kx    The order of the spline in x (use 3).
 * @param[out] tx   Array of knot positions in x (see note on size).
 * @param[in] ky    The order of the spline in y (use 3).
 * @param[out] ty   Array of knot positions in y (see note on size).
 * @param[out] nx   The number of knot positions generated in x.
 * @param[out] ny   The number of knot positions generated in y.
 * @param[out] c    The spline coefficients (see note on size).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_spline_set_up_f(int num_x, const float* x, int num_y,
        const float* y, const float* data, int kx, float* tx,
        int ky, float* ty, int* nx, int* ny, float* c);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPLINE_SET_UP_H_ */
