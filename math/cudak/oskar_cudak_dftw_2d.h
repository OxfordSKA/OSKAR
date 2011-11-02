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

#ifndef OSKAR_CUDAK_DFTW_2D_H_
#define OSKAR_CUDAK_DFTW_2D_H_

/**
 * @file oskar_cudak_dftw_2d.h
 */

#include "oskar_global.h"

/**
 * @brief
 * CUDA kernel to generate un-normalised 2D DFT weights (single precision).
 *
 * @details
 * This CUDA kernel produces the complex 2D DFT weights for the
 * given inputs and output, and stores them in device memory.
 *
 * Each thread generates the complex weight for a single input.
 *
 * Note: input x,y positions for spatial DFT weights should be in wavenumber
 * units.
 *
 * @param[in] n_in     Number of input points.
 * @param[in] x_in     Array of input x positions.
 * @param[in] y_in     Array of input y positions.
 * @param[in] x_out    Output 1/x position.
 * @param[in] y_out    Output 1/y position.
 * @param[out] weights Vector of complex DFT weights (length n_in).
 */
__global__
void oskar_cudak_dftw_2d_f(const int n_in, const float* x_in,
        const float* y_in, const float x_out, const float y_out,
        float2* weights);

/**
 * @brief
 * CUDA kernel to generate un-normalised 2D DFT weights (double precision).
 *
 * @details
 * This CUDA kernel produces the complex 2D DFT weights for the
 * given inputs and output, and stores them in device memory.
 *
 * Each thread generates the complex weight for a single input.
 *
 * Note: input x,y positions for spatial DFT weights should be in wavenumber
 * units.
 *
 * @param[in] n_in     Number of input points.
 * @param[in] x_in     Array of input x positions.
 * @param[in] y_in     Array of input y positions.
 * @param[in] x_out    Output 1/x position.
 * @param[in] y_out    Output 1/y position.
 * @param[out] weights Vector of complex DFT weights (length n_in).
 */
__global__
void oskar_cudak_dftw_2d_d(const int n_in, const double* x_in,
        const double* y_in, const double x_out, const double y_out,
        double2* weights);

#endif // OSKAR_CUDAK_DFTW_2D_H_
