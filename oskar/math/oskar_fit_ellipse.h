/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#ifndef OSKAR_FIT_ELLIPSE_H_
#define OSKAR_FIT_ELLIPSE_H_

/**
 * @file oskar_fit_ellipse.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Fits an ellipse to the specified points (single precision).
 *
 * @details
 * This function fits a 2D ellipse to the locus of points in the arrays
 * \p x and \p y.
 *
 * There must be at least 5 points in the arrays for this function to work.
 *
 * @param[out] major               Major axis of the fitted ellipse.
 * @param[out] minor               Minor axis of the fitted ellipse.
 * @param[out] position_angle_rad  Position angle of the ellipse, in radians.
 * @param[in] num_points           Number of points to fit.
 * @param[in] x                    Array of point x coordinates.
 * @param[in] y                    Array of point y coordinates.
 * @param[in] work1_5_num_points   Work array of length 5 * num_points.
 * @param[in] work2_5_num_points   Work array of length 5 * num_points.
 * @param[in,out] status           Status return code.
 */
OSKAR_EXPORT
void oskar_fit_ellipse_f(float* major, float* minor,
        float* position_angle_rad, int num_points, const float* x,
        const float* y, float* work1_5_num_points, float* work2_5_num_points,
        int* status);

/**
 * @brief
 * Fits an ellipse to the specified points (double precision).
 *
 * @details
 * This function fits a 2D ellipse to the locus of points in the arrays
 * \p x and \p y.
 *
 * There must be at least 5 points in the arrays for this function to work.
 *
 * @param[out] major               Major axis of the fitted ellipse.
 * @param[out] minor               Minor axis of the fitted ellipse.
 * @param[out] position_angle_rad  Position angle of the ellipse, in radians.
 * @param[in] num_points           Number of points to fit.
 * @param[in] x                    Array of point x coordinates.
 * @param[in] y                    Array of point y coordinates.
 * @param[in] work1_5_num_points   Work array of length 5 * num_points.
 * @param[in] work2_5_num_points   Work array of length 5 * num_points.
 * @param[in,out] status           Status return code.
 */
OSKAR_EXPORT
void oskar_fit_ellipse_d(double* major, double* minor,
        double* position_angle_rad, int num_points, const double* x,
        const double* y, double* work1_5_num_points, double* work2_5_num_points,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FIT_ELLIPSE_H_ */
