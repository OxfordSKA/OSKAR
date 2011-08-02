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

#ifndef OSKAR_CUDA_CORRELATOR_SCALAR_STANDALONE_H_
#define OSKAR_CUDA_CORRELATOR_SCALAR_STANDALONE_H_

/**
 * @file oskar_cuda_correlator_scalar_standalone.h
 */


#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief
 * Computes complex visibilities (single precision).
 *
 * @details
 *
 * @param[in] num_antennas Number of antennas or stations.
 */
DllExport
int oskar_cudaf_correlator_scalar_standalone(const int num_antennas,
        const float* antenna_x, const float* antenna_y, const float* antenna_z,
        const int num_sources, const float* source_l, const float* source_m,
        const float* b_sqrt, const float* e, const float ra0, const float dec0,
        const float lst0, const int nsdt, const float std, const float k,
        const float lambda_bandwidth, float* vis);


/**
 * @brief
 * Allocates memory for use with
 *
 * @details
 *
 * @param[in] num_antennas The number of antennas/ stations to be correlated.
 * @param[in] num_sources  The number of sources to be correlated.
 */
DllExport
int oskar_cudaf_correlator_scalar_allocate_memory(const unsigned num_antennas,
        const unsigned num_sources, float* d_antenna_x, float* d_antenna_y,
        float* d_antenna_z, float* d_source_l, float* d_source_m,
        float* d_source_n, float* d_eb, float* d_work);





#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_CORRELATOR_SCALAR_STANDALONE_H_
