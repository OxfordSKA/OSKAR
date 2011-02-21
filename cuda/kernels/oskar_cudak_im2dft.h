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

#ifndef OSKAR_CUDAK_IM2DFT_H_
#define OSKAR_CUDAK_IM2DFT_H_

/**
 * @file oskar_cudak_im2dft.h
 */

#include "cuda/CudaEclipse.h"

/**
 * @brief
 * CUDA kernel to compute an image using a simple 2D DFT.
 *
 * @details
 * This CUDA kernel computes a real image from a set of complex visibilities,
 * using a 2D Direct Fourier Transform (DFT).
 *
 * The computed image is returned in the \p image array, which
 * must be pre-sized to length \p np.
 *
 * Each thread evaluates a single pixel of the image, which is assumed to be
 * completely real: conjugated copies of the visibilities should therefore NOT
 * be supplied to this kernel.
 *
 * The kernel requires (4 * maxVisPerBlock) * sizeof(float) bytes of
 * shared memory.
 *
 * @param[in] nv No. of independent visibilities (excluding Hermitian copy).
 * @param[in] u Array of visibility u coordinates in wavelengths (length nv).
 * @param[in] v Array of visibility v coordinates in wavelengths (length nv).
 * @param[in] vis Array of complex visibilities (length nv; see note, above).
 * @param[in] np The number of pixels in the image.
 * @param[in] pl The pixel l-positions on the orthographic tangent plane.
 * @param[in] pm The pixel m-positions on the orthographic tangent plane.
 * @param[in] maxVisPerBlock Maximum visibilities per block (multiple of 16).
 * @param[out] image The computed image (see note, above).
 */
__global__
void oskar_cudak_im2dft(int nv, const float* u, const float* v,
        const float2* vis, const int np, const float* pl, const float* pm,
        const int maxVisPerBlock, float* image);

#endif // OSKAR_CUDAK_IM2DFT_H_
