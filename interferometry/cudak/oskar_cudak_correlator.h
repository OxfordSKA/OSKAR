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

#ifndef OSKAR_CUDAK_CORRELATOR_H_
#define OSKAR_CUDAK_CORRELATOR_H_

/**
 * @file oskar_cudak_correlator.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 *
 * @details
 *
 * @param[in] ns               Number of sources.
 * @param[in] na               Number of stations.
 * @param[in] u                Station u-coordinates multipled by the wavenumber.
 * @param[in] v                Station v-coordinates multipled by the wavenumber.
 * @param[in] l                Distance from phase centre for each source.
 * @param[in] m                Distance from phase centre for each source.
 * @param[in] lambda_bandwidth Wavelength (m) times bandwidth (Hz).
 * @param[in,out] vis          Modified output complex visibilities.
 */
__global__
void oskar_cudak_correlator_f(const int ns, const int na,
        const float4c* jones, const float* source_I, const float* source_Q,
        const float* source_U, const float* source_V, const float* u,
        const float* v, const float* l, const float* m,
        const float lambda_bandwidth, float4c* vis);

/**
 * @brief
 *
 * @details
 *
 * @param[in] ns               Number of sources.
 * @param[in] na               Number of stations.
 * @param[in] u                Station u-coordinates multipled by the wavenumber.
 * @param[in] v                Station v-coordinates multipled by the wavenumber.
 * @param[in] l                Distance from phase centre for each source.
 * @param[in] m                Distance from phase centre for each source.
 * @param[in] lambda_bandwidth Wavelength (m) times bandwidth (Hz).
 * @param[in,out] vis          Modified output complex visibilities.
 */
__global__
void oskar_cudak_correlator_d(const int ns, const int na,
        const double4c* jones, const double* source_I, const double* source_Q,
        const double* source_U, const double* source_V, const double* u,
        const double* v, const double* l, const double* m,
        const double lambda_bandwidth, double4c* vis);

#endif // OSKAR_CUDAK_CORRELATOR_H_
