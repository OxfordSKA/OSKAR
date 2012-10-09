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

#ifndef OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_CUDA_H_
#define OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_CUDA_H_

/**
 * @file oskar_evaluate_element_weights_dft_cuda.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates geometric DFT beamforming weights using CUDA (single precision).
 *
 * @details
 * This function produces the complex DFT beamforming weights for the
 * given element positions and beam direction.
 *
 * Note that all element positions must be in radians (i.e. pre-multiplied
 * by the wavenumber).
 *
 * Note that all pointers refer to device memory.
 *
 * @param[out] weights      Array of complex DFT weights (length num_elements).
 * @param[in] num_elements  Number of antenna elements.
 * @param[in] d_x           Input element x positions, in radians.
 * @param[in] d_y           Input element y positions, in radians.
 * @param[in] d_y           Input element z positions, in radians.
 * @param[in] x_beam        Beam direction cosine x-component.
 * @param[in] y_beam        Beam direction cosine y-component.
 * @param[in] z_beam        Beam direction cosine z-component.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_dft_cuda_f(float2* d_weights,
        int num_elements, const float* d_x, const float* d_y,
        const float* d_z, float x_beam, float y_beam, float z_beam);

/**
 * @brief
 * Generates geometric DFT beamforming weights using CUDA (double precision).
 *
 * @details
 * This function produces the complex DFT beamforming weights for the
 * given element positions and beam direction.
 *
 * Note that all element positions must be in radians (i.e. pre-multiplied
 * by the wavenumber).
 *
 * Note that all pointers refer to device memory.
 *
 * @param[out] weights      Array of complex DFT weights (length num_elements).
 * @param[in] num_elements  Number of antenna elements.
 * @param[in] d_x           Input element x positions, in radians.
 * @param[in] d_y           Input element y positions, in radians.
 * @param[in] d_y           Input element z positions, in radians.
 * @param[in] x_beam        Beam direction cosine x-component.
 * @param[in] y_beam        Beam direction cosine y-component.
 * @param[in] z_beam        Beam direction cosine z-component.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_dft_cuda_d(double2* d_weights,
        int num_elements, const double* d_x, const double* d_y,
        const double* d_z, double x_beam, double y_beam, double z_beam);

#ifdef __CUDACC__

/**
 * @brief
 * CUDA kernel to generate un-normalised 3D DFT weights (single precision).
 *
 * @details
 * This CUDA kernel produces the complex 3D DFT weights for the
 * given inputs and output, and stores them in device memory.
 *
 * Each thread generates the complex weight for a single input.
 *
 * Note: input positions for spatial DFT weights should be in radians.
 *
 * @param[out] weights Vector of complex DFT weights (length n_in).
 * @param[in] n_in     Number of input points.
 * @param[in] x_in     Array of input x positions.
 * @param[in] y_in     Array of input y positions.
 * @param[in] z_in     Array of input z positions.
 * @param[in] x_out    Output 1/x position.
 * @param[in] y_out    Output 1/y position.
 * @param[in] z_out    Output 1/z position.
 */
OSKAR_EXPORT
__global__
void oskar_evaluate_element_weights_dft_cudak_f(float2* weights,
        const int n_in, const float* x_in, const float* y_in,
        const float* z_in, const float x_out, const float y_out,
        const float z_out);

/**
 * @brief
 * CUDA kernel to generate un-normalised 3D DFT weights (double precision).
 *
 * @details
 * This CUDA kernel produces the complex 3D DFT weights for the
 * given inputs and output, and stores them in device memory.
 *
 * Each thread generates the complex weight for a single input.
 *
 * Note: input positions for spatial DFT weights should be in radians.
 *
 * @param[out] weights Vector of complex DFT weights (length n_in).
 * @param[in] n_in     Number of input points.
 * @param[in] x_in     Array of input x positions.
 * @param[in] y_in     Array of input y positions.
 * @param[in] z_in     Array of input z positions.
 * @param[in] x_out    Output 1/x position.
 * @param[in] y_out    Output 1/y position.
 * @param[in] z_out    Output 1/z position.
 */
OSKAR_EXPORT
__global__
void oskar_evaluate_element_weights_dft_cudak_d(double2* weights,
        const int n_in, const double* x_in, const double* y_in,
        const double* z_in, const double x_out, const double y_out,
        const double z_out);

#endif

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_ELEMENT_WEIGHTS_DFT_CUDA_H_ */
