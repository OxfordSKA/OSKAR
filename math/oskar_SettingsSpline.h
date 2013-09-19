/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_SETTINGS_SPLINE_H_
#define OSKAR_SETTINGS_SPLINE_H_

/**
 * @file oskar_SettingsSpline.h
 */

/**
 * @struct oskar_SettingsSpline
 *
 * @brief Structure to hold settings for spline surface fitting.
 *
 * @details
 * The structure holds settings used to parameterise the spline surface fitting
 * procedure.
 */
struct oskar_SettingsSpline
{
    double eps_float; /**< Epsilon for single precision fit. */
    double eps_double; /**< Epsilon for double precision fit. */
    double smoothness_factor_override; /**< Value to use for smoothness factor, if not searching. */
    int search_for_best_fit; /**< If true, then search for best fit using average fractional error. */
    double average_fractional_error; /**< Target average fractional error. */
    double average_fractional_error_factor_increase; /**< In case of fitting failure, factor by which to increase allowed average fractional error. */
};
typedef struct oskar_SettingsSpline oskar_SettingsSpline;

#endif /* OSKAR_SETTINGS_SPLINE_H_ */
