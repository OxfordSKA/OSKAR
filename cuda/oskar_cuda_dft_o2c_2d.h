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

#ifndef OSKAR_CUDA_DFT_O2C_2D_H_
#define OSKAR_CUDA_DFT_O2C_2D_H_

/**
 * @file oskar_cuda_dft_o2c_2d.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * CUDA kernel to perform a 2D real-to-complex single-precision DFT, where
 * all input signals are implicitly of amplitude 1.0.
 *
 * @details
 * Computes a 2D real-to-complex DFT using CUDA, assuming all input signals
 * are implicitly of amplitude 1.0.
 *
 * Note that all pointers are device pointers, and must not be dereferenced
 * in host code.
 *
 * This function must be supplied with the input x- and y-positions, the
 * output x- and y-positions, and the phase centre. The input positions must
 * be pre-multiplied by a factor k (= 2pi / lambda), and the output positions
 * and phase centre are direction cosines.
 *
 * The computed points are returned in the \p output array, which must be
 * pre-sized to length 2 * n_out. The values in the \p output array are
 * alternate (real, imag) pairs for each output position.
 *
 * @param[in] n_in         Number of input points.
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] n_out        Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[in] x_centre     Phase centre 1/x position.
 * @param[in] y_centre     Phase centre 1/y position.
 * @param[in] work         Work array (size 2 * n_in).
 * @param[out] output      Array of computed output points (size 2 * n_out).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
OSKAR_EXPORT
int oskar_cudaf_dft_o2c_2d(int n_in, const float* x_in, const float* y_in,
        int n_out, const float* x_out, const float* y_out, float x_centre,
        float y_centre, float* work, float* output);

/**
 * @brief
 * CUDA kernel to perform a 2D real-to-complex double-precision DFT, where
 * all input signals are implicitly of amplitude 1.0.
 *
 * @details
 * Computes a 2D real-to-complex DFT using CUDA, assuming all input signals
 * are implicitly of amplitude 1.0.
 *
 * Note that all pointers are device pointers, and must not be dereferenced
 * in host code.
 *
 * This function must be supplied with the input x- and y-positions, the
 * output x- and y-positions, and the phase centre. The input positions must
 * be pre-multiplied by a factor k (= 2pi / lambda), and the output positions
 * and phase centre are direction cosines.
 *
 * The computed points are returned in the \p output array, which must be
 * pre-sized to length 2 * n_out. The values in the \p output array are
 * alternate (real, imag) pairs for each output position.
 *
 * @param[in] n_in         Number of input points.
 * @param[in] x_in         Array of input x positions.
 * @param[in] y_in         Array of input y positions.
 * @param[in] n_out        Number of output points.
 * @param[in] x_out        Array of output 1/x positions.
 * @param[in] y_out        Array of output 1/y positions.
 * @param[in] x_centre     Phase centre 1/x position.
 * @param[in] y_centre     Phase centre 1/y position.
 * @param[in] work         Work array (size 2 * n_in).
 * @param[out] output      Array of computed output points (size 2 * n_out).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
OSKAR_EXPORT
int oskar_cudad_dft_o2c_2d(int n_in, const double* x_in, const double* y_in,
        int n_out, const double* x_out, const double* y_out, double x_centre,
        double y_centre, double* work, double* output);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_DFT_O2C_2D_H_
