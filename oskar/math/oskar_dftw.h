/*
 * Copyright (c) 2017, The University of Oxford
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

#ifndef OSKAR_DFTW_H_
#define OSKAR_DFTW_H_

/**
 * @file oskar_dftw.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to perform a DFT using supplied weights.
 *
 * @details
 * This function performs a DFT using the supplied weights array.
 *
 * The transform may be either 2D or 3D. If either \p z_in or \p z_out
 * is NULL on input, the transform will be done in 2D.
 *
 * The wavelength used to compute the supplied wavenumber must be in the
 * same units as the input positions (e.g. metres).
 *
 * If the \p data parameter is NULL, then all input values will
 * be implicitly assumed to be real, with value 1.0. If not NULL,
 * the \p data array must be complex and of size \p num_out * \p num_in.
 * It is accessed in such a way that the output dimension must be the
 * fastest varying.
 *
 * The computed points are returned in the \p output array.
 * These are the complex values for each output position.
 *
 * @param[in] num_in       Number of input points.
 * @param[in] wavenumber   Wavenumber (2 pi / wavelength).
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] z_in         Array of input z positions.
 * @param[in] weights_in   Array of complex DFT weights.
 * @param[in] num_out      Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[in] z_out        Array of output 1/z positions.
 * @param[out] data        Input data (see note, above).
 * @param[out] output      Array of computed output points (see note, above).
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_dftw(
        int num_in,
        double wavenumber,
        const oskar_Mem* x_in,
        const oskar_Mem* y_in,
        const oskar_Mem* z_in,
        const oskar_Mem* weights_in,
        int num_out,
        const oskar_Mem* x_out,
        const oskar_Mem* y_out,
        const oskar_Mem* z_out,
        const oskar_Mem* data,
        oskar_Mem* output,
        int* status);

#ifdef __cplusplus
}
#endif

#endif
