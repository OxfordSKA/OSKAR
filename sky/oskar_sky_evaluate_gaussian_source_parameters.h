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

#ifndef OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_
#define OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_

/**
 * @file oskar_sky_evaluate_gaussian_source_parameters.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates Gaussian parameters for extended sources.
 *
 * @details
 * see http://en.wikipedia.org/wiki/Gaussian_function
 *
 * This is achieved by projecting the source ellipse as defined on the sky
 * to the observation l,m plane.
 *
 * - Six points are evaluated on the circumference of the ellipse which defines
 *   the gaussian source
 * - These points are projected to the l,m plane
 * - Points on the l,m plane are then used to fit a new ellipse which defines
 *   the l,m plane gaussian function of the source
 * - 2D Gaussian parameters are evaluated from the fit of the l,m plane ellipse.
 *
 * Fitting of the ellipse on the l,m plane is carried out by oskar_fit_ellipse()
 * which uses the LAPACK routines (D|S)GETRS and (D|S)GETRF to perform the
 * fitting.
 *
 * TODO better description of how this works... (see MATLAB code)
 *
 * @param[in,out] sky      Sky model to update.
 * @param[in] zero_failed_sources If set, zero amplitude of sources
 *                                where the Gaussian solution fails.
 * @param[in] ra0          Right ascension of the observation phase centre.
 * @param[in] dec0         Declination of the observation phase centre.
 * @param[out] num_failed  The number of sources where Gaussian fitting failed.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_evaluate_gaussian_source_parameters(oskar_Sky* sky,
        int zero_failed_sources, double ra0, double dec0, int* num_failed,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_ */
