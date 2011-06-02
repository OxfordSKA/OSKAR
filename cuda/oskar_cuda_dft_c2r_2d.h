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

#ifndef OSKAR_CUDA_DFT_C2R_2D_H_
#define OSKAR_CUDA_DFT_C2R_2D_H_

/**
 * @file oskar_cuda_dft_c2r_2d.h
 */

#include "oskar_cuda_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * CUDA wrapper to perform a 2D complex-to-real single-precision DFT.
 *
 * @details
 * Computes a real output from a set of complex input data, using CUDA to
 * evaluate a 2D Direct Fourier Transform (DFT).
 *
 * Note that all pointers are device pointers, and must not be dereferenced
 * in host code.
 *
 * This function must be supplied with the input x- and y-positions, and the
 * output x- and y-positions. The input positions must be pre-multiplied by a
 * factor k (= 2pi / lambda), and the output positions are direction cosines.
 *
 * The fastest-varying dimension in the output array is along x. The output is
 * assumed to be completely real, so the conjugate copy of the input data
 * should not be supplied.
 *
 * @param[in] n_in         Number of input points.
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] data_in      Array of complex input data (length 2 * n_in).
 * @param[in] n_out        Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[out] output      Array of computed output points.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
DllExport
int oskar_cudaf_dft_c2r_2d(int n_in, const float* x_in, const float* y_in,
        const float* data_in, int n_out, const float* x_out,
        const float* y_out, float* output);

/**
 * @brief
 * CUDA wrapper to perform a 2D complex-to-real double-precision DFT.
 *
 * @details
 * Computes a real output from a set of complex input data, using CUDA to
 * evaluate a 2D Direct Fourier Transform (DFT).
 *
 * Note that all pointers are device pointers, and must not be dereferenced
 * in host code.
 *
 * This function must be supplied with the input x- and y-positions, and the
 * output x- and y-positions. The input positions must be pre-multiplied by a
 * factor k (= 2pi / lambda), and the output positions are direction cosines.
 *
 * The fastest-varying dimension in the output array is along x. The output is
 * assumed to be completely real, so the conjugate copy of the input data
 * should not be supplied.
 *
 * @param[in] n_in         Number of input points.
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] data_in      Array of complex input data (length 2 * n_in).
 * @param[in] n_out        Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[out] output      Array of computed output points.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
DllExport
int oskar_cudad_dft_c2r_2d(int n_in, const double* x_in, const double* y_in,
        const double* data_in, int n_out, const double* x_out,
        const double* y_out, double* output);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_DFT_C2R_2D_H_
