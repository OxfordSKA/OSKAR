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

#ifndef OSKAR_BF_CUDAK_ANT_SIG_ISO_2D_H_
#define OSKAR_BF_CUDAK_ANT_SIG_ISO_2D_H_

/**
 * @file oskar_bf_cudak_ant_sig_iso_2d.h
 */

#include "utility/oskar_util_cuda_eclipse.h"

/**
 * @brief
 * CUDA kernel to compute antenna signals for beamforming simulation.
 *
 * @details
 * This CUDA kernel evaluates the antenna signals for the given source and
 * antenna positions. It requires (8 * number_of_threads_per_block) bytes
 * of shared memory to be preallocated by the caller.
 *
 * Each thread evaluates the signal for a single antenna, looping over
 * all the sources.
 *
 * The cosine and sine of the source azimuths, and the cosine
 * of the elevations, must be given as triplets in the \p strig array:
 *
 * strig.x = {cosine azimuth}
 * strig.y = {sine azimuth}
 * strig.z = {cosine elevation}
 *
 * The computed antenna signals are returned in the \p signals array, which
 * must be pre-sized to length 2*na. The values in the \p signals array
 * are alternate (real, imag) pairs for each antenna.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] strig The cosine and sine of the source coordinates.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
__global__
void oskar_bf_cudakf_ant_sig_iso_2d(const int na, const float* ax,
		const float* ay, const int ns, const float* samp, const float3* strig,
		const float k, const int maxSourcesPerBlock, float2* signals);

#endif // OSKAR_BF_CUDAK_ANT_SIG_ISO_2D_H_
