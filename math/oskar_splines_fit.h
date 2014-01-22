/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_SPLINES_FIT_H_
#define OSKAR_SPLINES_FIT_H_

/**
 * @file oskar_splines_fit.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_SettingsSpline.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes spline data from a list of data points.
 *
 * @details
 * This function constructs splines from a list of data points.
 *
 * @param[in,out] spline         Pointer to spline data structure.
 * @param[in]     num_points     Number of data points in all arrays.
 * @param[in]     x              Array of x positions.
 * @param[in]     y              Array of y positions.
 * @param[in]     z              Array of data points.
 * @param[in]     w              Array of data point weights.
 * @param[in]     search         If set, use supplied average fractional error.
 * @param[in,out] avg_frac_error On entry, the target average fractional error;
 *                               on exit, the achieved average fractional error.
 * @param[in]     inc_factor     Factor by which to increase average fractional
 *                               error on failure.
 * @param[in]     smooth_factor  A user-supplied smoothing factor.
 *                               Ignored if \p search parameter is clear.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_splines_fit(oskar_Splines* spline, int num_points, oskar_Mem* x,
        oskar_Mem* y, const oskar_Mem* z, const oskar_Mem* w, int search,
        double* avg_frac_err, double inc_factor, double user_s,
        double eps_float, double eps_double, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPLINES_FIT_H_ */
