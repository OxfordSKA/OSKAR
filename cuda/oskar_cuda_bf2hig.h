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

#ifndef OSKAR_CUDA_BF2HIG_H_
#define OSKAR_CUDA_BF2HIG_H_

/**
 * @file oskar_cuda_bf2hig.h
 */

#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes beams using CUDA.
 *
 * @details
 * Computes beams using CUDA.
 *
 * The computed beams are returned in the \p beams array, which
 * must be pre-sized to length 2*nb. The values in the \p beams array
 * are alternate (real, imag) pairs for each beam.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of sources.
 * @param[in] samp The source amplitudes.
 * @param[in] slon The source longitude coordinates in radians.
 * @param[in] slat The source latitude coordinates in radians.
 * @param[in] nb The number of beams to form.
 * @param[in] blon The source longitude coordinates in radians.
 * @param[in] blat The source latitude coordinates in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] beams The complex vector of output beams (length nb).
 */
DllExport
void oskar_cudaf_bf2hig(int na, const float* ax, const float* ay,
        int ns, const float* samp, const float* slon, const float* slat,
        int nb, const float* blon, const float* blat, float k,
        float* beams);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_BF2HIG_H_
