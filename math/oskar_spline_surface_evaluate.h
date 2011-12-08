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

#ifndef OSKAR_SPLINE_SURFACE_EVALUATE_H_
#define OSKAR_SPLINE_SURFACE_EVALUATE_H_

/**
 * @file oskar_spline_surface_evaluate.h
 */

#include "oskar_global.h"
#include "math/oskar_SplineData.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a surface fitted by splines, at the given positions.
 *
 * @details
 * This function evaluates a surface fitted by splines.
 *
 * It calls the FORTRAN function bispev from the DIERCKX library to perform the
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
 * @param[in] tx   Array of knot positions in x.
 * @param[in] nx   The number of knot positions generated in x.
 * @param[in] ty   Array of knot positions in y (see note on size).
 * @param[in] ny   The number of knot positions generated in y.
 * @param[in] c    The spline coefficients.
 * @param[in] kx   The order of the spline in x (use 3).
 * @param[in] ky   The order of the spline in y (use 3).
 * @param[in] num_points The number of points at which to evaluate the surface.
 * @param[in] x    Array of x positions.
 * @param[in] y    Array of x positions.
 * @param[out] output Array of evaluated surface values.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_spline_surface_evaluate_f(const float* tx, int nx, const float* ty,
		int ny, const float* c, int kx, int ky, int num_points, const float* x,
		const float* y, float* output);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPLINE_SURFACE_EVALUATE_H_ */
