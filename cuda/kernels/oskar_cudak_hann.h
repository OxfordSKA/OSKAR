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

#ifndef OSKAR_CUDAK_HANN_H_
#define OSKAR_CUDAK_HANN_H_

/**
 * @file oskar_cudak_hann.h
 */

#include "cuda/CudaEclipse.h"


/**
 * @brief
 * CUDA kernel to apply a hann apodisation into beamforing weights.
 *
 * @details
 * http://mathworld.wolfram.com/HanningFunction.html
 *
 * weights matrix dimensions = weights[b][a]
 *
 * @param[in] na    Number of antennas.
 * @param[in] ax    Antenna x positions (metres).
 * @param[in] ay    Antenna y positions (metres).
 * @param[in] nb    Number of beam directions.
 * @param[in] fwhm  Full with at half maximum of the hann window.
 * @param[in/out] weights Weights matrix to which to apply apodisation.
 */
__global__
void oskar_cudak_hann(const unsigned na, const float * ax, const float * ay,
        const unsigned nb, const float fwhm, float2 * weights);

#endif // OSKAR_CUDAK_HANN_H_
