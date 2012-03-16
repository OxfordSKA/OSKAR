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


#ifndef OSKAR_EVALUATE_GRIDDING_KERNEL_H_
#define OSKAR_EVALUATE_GRIDDING_KERNEL_H_

#include "imaging/fft/oskar_GridKernel.h"
#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_initialise_kernel_d(const double radius, oskar_GridKernel_d* kernel);

/**
 * @brief
 * Generates a pill-box gridding convolution kernel (double precision)
 *
 * @details
 * Note: This function allocates memory for the kernel internally.
 *
 * @param[out] kernel       Pointer to a structure holding the gridding kernel.
 */
OSKAR_EXPORT
void oskar_evaluate_pillbox_d(oskar_GridKernel_d* kernel);

/**
 * @brief
 * Generates a exponential times sinc gridding convolution kernel
 * (double precision)
 *
 * @details
 * Note: This function allocates memory for the kernel internally.
 *
 * @param[out] kernel       Pointer to a structure holding the gridding kernel.
 */
OSKAR_EXPORT
void oskar_evaluate_exp_sinc_d(oskar_GridKernel_d* kernel);


/**
 * @brief
 * Generates a Spherodial wave function convolution kernel (double precision)
 *
 * @details
 * Note: This function allocates memory for the kernel internally.
 *
 * @param[out] kernel       Pointer to a structure holding the gridding kernel.
 */
OSKAR_EXPORT
void oskar_evaluate_spheroidal_d(oskar_GridKernel_d* kernel);


/**
 * @brief
 * Evaluates the amplitude of a spheroid wave function at a particular
 * value of ...
 *       see AIPS/<VERSION>/APL/SUB/NOTST/SPHFN.FOR
 */
OSKAR_EXPORT
void spheroidal_d(int iAlpha, int iflag, double eta, double* value);


#ifdef __cplusplus
}
#endif

#endif // OSKAR_EVALUATE_GRIDDING_KERNEL_H_
