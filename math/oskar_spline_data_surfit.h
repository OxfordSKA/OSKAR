/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_SPLINE_DATA_SURFIT_H_
#define OSKAR_SPLINE_DATA_SURFIT_H_

/**
 * @file oskar_spline_data_surfit.h
 */

#include "oskar_global.h"
#include "math/oskar_SplineData.h"
#include "math/oskar_SettingsSpline.h"
#include "utility/oskar_Log.h"

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
 * @param[in,out] spline        Pointer to spline data structure.
 * @param[in,out] log           Pointer to log structure to use.
 * @param[in]     num_points    Number of data points in all arrays.
 * @param[in]     x             Array of x positions.
 * @param[in]     y             Array of y positions.
 * @param[in]     z             Array of data points.
 * @param[in]     w             Array of data point weights.
 * @param[in]     settings      Fitting parameters.
 * @param[in]     surface_name  Name of fitted surface (for log).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_spline_data_surfit(oskar_SplineData* spline, oskar_Log* log,
        int num_points, oskar_Mem* x, oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* w, const oskar_SettingsSpline* settings,
        const char* surface_name);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPLINE_DATA_SURFIT_H_ */
