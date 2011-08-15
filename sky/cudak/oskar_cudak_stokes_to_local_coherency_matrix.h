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

#ifndef OSKAR_CUDAK_STOKES_TO_LOCAL_COHERENCY_MATRIX_H_
#define OSKAR_CUDAK_STOKES_TO_LOCAL_COHERENCY_MATRIX_H_

/**
 * @file oskar_cudak_stokes_to_local_coherency_matrix.h
 */

#include "utility/oskar_cuda_eclipse.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 * CUDA kernel to compute average source local coherency matrix
 * (single precision).
 *
 * @details
 *
 * The directions are:
 * <li> l is parallel to x (pointing East), </li>
 * <li> m is parallel to y (pointing North), </li>
 * <li> n is parallel to z (pointing to the zenith). </li>
 *
 * @param[in] ns The number of source positions.
 * @param[in] ra The source Right Ascensions in radians.
 * @param[in] dec The source Declinations in radians.
 * @param[in] cosLat The cosine of the geographic latitude.
 * @param[in] sinLat The sine of the geographic latitude.
 * @param[in] lst The local sidereal time in radians.
 * @param[in] l The source l-coordinates (see above).
 * @param[in] m The source m-coordinates (see above).
 * @param[in] n The source n-coordinates (see above).
 */
__global__
void oskar_cudak_stokes_to_local_coherency_matrix_f(int ns, const float* ra,
        const float* dec, const float* stokes_I, const float* stokes_Q,
        const float* stokes_U, const float* stokes_V, float cos_lat,
        float sin_lat, float lst, float4c* coherency_matrix);

/**
 * @brief
 * CUDA kernel to compute average source local coherency matrix
 * (double precision).
 *
 * @details
 *
 * The directions are:
 * <li> l is parallel to x (pointing East), </li>
 * <li> m is parallel to y (pointing North), </li>
 * <li> n is parallel to z (pointing to the zenith). </li>
 *
 * @param[in] ns The number of source positions.
 * @param[in] ra The source Right Ascensions in radians.
 * @param[in] dec The source Declinations in radians.
 * @param[in] cosLat The cosine of the geographic latitude.
 * @param[in] sinLat The sine of the geographic latitude.
 * @param[in] lst The local sidereal time in radians.
 * @param[out] l The source l-coordinates (see above).
 * @param[out] m The source m-coordinates (see above).
 * @param[out] n The source n-coordinates (see above).
 */
__global__
void oskar_cudak_stokes_to_local_coherency_matrix_d(int ns, const double* ra,
        const double* dec, const double* stokes_I, const double* stokes_Q,
        const double* stokes_U, const double* stokes_V, double cos_lat,
        double sin_lat, double lst, double4c* coherency_matrix);

#endif // OSKAR_CUDAK_STOKES_TO_LOCAL_COHERENCY_MATRIX_H_
