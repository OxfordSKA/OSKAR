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

#ifndef OSKAR_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_
#define OSKAR_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_

/**
 * @file oskar_evaluate_gaussian_source_parameters.h
 */

#include "oskar_global.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Mem.h"

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
 * to the l,m observation l,m plane.
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
 * FIXME if zero failed sources stays in the code, it will need to be modified
 * to also take Q,U,V
 *
 *
 * @param log             OSKAR message log object
 * @param num_sources     Number of sources (length of source arrays)
 * @param gaussian_a      Gaussian parameter array
 * @param gaussian_b      Gaussian parameter array
 * @param gaussian_c      Gaussian parameter array
 * @param FWHM_major      Source major axis FWHM array
 * @param FWHM_minor      Source minor axis FWHM array
 * @param position_angle  Source position angle array
 * @param RA              Source right ascension array
 * @param Dec             Source declination array
 * @param zero_failed_sources Bool to set to zero the amplitude of sources
 *                        where the Gaussian solution fails
 * @param I               Unpolarised source flux density.
 * @param ra0             Right ascension of the observation phase centre
 * @param dec0            Declination of the observation phase centre
 *
 * @return An error code
 */
OSKAR_EXPORT
int oskar_evaluate_gaussian_source_parameters(oskar_Log* log, int num_sources,
        oskar_Mem* gaussian_a, oskar_Mem* gaussian_b, oskar_Mem* gaussian_c,
        const oskar_Mem* FWHM_major, const oskar_Mem* FWHM_minor,
        const oskar_Mem* position_angle, const oskar_Mem* RA,
        const oskar_Mem* Dec, int zero_failed_sources, oskar_Mem* I,
        double ra0, double dec0);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_ */
