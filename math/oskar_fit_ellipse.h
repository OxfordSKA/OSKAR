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

#ifndef OSKAR_FIT_ELLIPSE_H_
#define OSKAR_FIT_ELLIPSE_H_

/**
 * @file oskar_fit_ellipse.h
 */

#include "oskar_global.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fits a 2D ellipse to the locus of points specified by the arrays @p x
 * and @p y.
 *
 * @details
 * Uses the LAPACK functions (D|S)GETRS and (D|S)GETRF
 *
 * @param log          OSKAR message log object
 * @param gauss_maj    Major axis of the fitted ellipse
 * @param gauss_min    Minor axis of the fitted ellipse
 * @param gauss_phi    Position angle of the fitted ellipse
 * @param num_points   Number of points to fit
 * @param x            Array of x coordinates to fit the ellipse to
 * @param y            Array of y coordinates to fit the ellispe to
 *
 * @return An error code
 */
OSKAR_EXPORT
int oskar_fit_ellipse(oskar_Log* log, double* gauss_maj, double* gauss_min,
        double* gauss_phi, int num_points, const oskar_Mem* x,
        const oskar_Mem* y);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FIT_ELLIPSE_H_ */
