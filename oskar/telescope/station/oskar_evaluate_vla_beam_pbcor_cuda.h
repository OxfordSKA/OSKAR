/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_EVALUATE_VLA_BEAM_PBCOR_CUDA_H_
#define OSKAR_EVALUATE_VLA_BEAM_PBCOR_CUDA_H_

/**
 * @file oskar_evaluate_vla_beam_pbcor_cuda.h
 */

#include <oskar_global.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (single precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_cuda_f(float* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3);

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (single precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_complex_cuda_f(float2* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3);

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (single precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_matrix_cuda_f(float4c* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3);

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (double precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_cuda_d(double* beam, int num_sources,
        const double* l, const double* m, const double freq_ghz,
        const double p1, const double p2, const double p3);

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (double precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_complex_cuda_d(double2* beam,
        int num_sources, const double* l, const double* m,
        const double freq_ghz, const double p1, const double p2,
        const double p3);

/**
 * @brief
 * Evaluates a scalar VLA dish beam, as implemented in the AIPS task PBCOR
 * (double precision).
 *
 * @details
 * This function evaluates a scalar VLA dish beam, as implemented in the AIPS
 * task PBCOR.
 *
 * See http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
 *
 * @param[out] beam          VLA beam evaluated at source positions.
 * @param[in]  num_sources   Number of sources at which to evaluate the beam.
 * @param[in]  l             Direction cosine of each source from phase centre.
 * @param[in]  m             Direction cosine of each source from phase centre.
 * @param[in]  freq_ghz      Current observing frequency in GHz.
 * @param[in]  p1            Value of PBPARM(3) for this frequency.
 * @param[in]  p2            Value of PBPARM(4) for this frequency.
 * @param[in]  p3            Value of PBPARM(5) for this frequency.
 */
OSKAR_EXPORT
void oskar_evaluate_vla_beam_pbcor_matrix_cuda_d(double4c* beam,
        int num_sources, const double* l, const double* m,
        const double freq_ghz, const double p1, const double p2,
        const double p3);


#ifdef __CUDACC__

__global__
void oskar_evaluate_vla_beam_pbcor_cudak_f(float* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3);

__global__
void oskar_evaluate_vla_beam_pbcor_complex_cudak_f(float2* beam,
        int num_sources, const float* l, const float* m, const float freq_ghz,
        const float p1, const float p2, const float p3);

__global__
void oskar_evaluate_vla_beam_pbcor_matrix_cudak_f(float4c* beam,
        int num_sources, const float* l, const float* m, const float freq_ghz,
        const float p1, const float p2, const float p3);


/* Double precision. */
__global__
void oskar_evaluate_vla_beam_pbcor_cudak_d(double* beam, int num_sources,
        const double* l, const double* m, const double freq_ghz,
        const double p1, const double p2, const double p3);

__global__
void oskar_evaluate_vla_beam_pbcor_complex_cudak_d(double2* beam,
        int num_sources, const double* l, const double* m,
        const double freq_ghz, const double p1, const double p2,
        const double p3);

__global__
void oskar_evaluate_vla_beam_pbcor_matrix_cudak_d(double4c* beam,
        int num_sources, const double* l, const double* m,
        const double freq_ghz, const double p1, const double p2,
        const double p3);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_VLA_BEAM_PBCOR_CUDA_H_ */
