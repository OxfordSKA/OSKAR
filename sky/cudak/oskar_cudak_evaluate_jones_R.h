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

#ifndef OSKAR_CUDAK_EVALUATE_JONES_R_H_
#define OSKAR_CUDAK_EVALUATE_JONES_R_H_

/**
 * @file oskar_cudak_evaluate_jones_R.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 * CUDA kernel to construct matrices for parallactic angle rotation and
 * conversion of linear Stokes parameters from equatorial to local horizontal
 * frame (single precision).
 *
 * @details
 * This CUDA kernel constructs a set of Jones matrices that will transform the
 * equatorial linear Stokes parameters into the local horizontal frame of the
 * station. This corresponds to a rotation by the parallactic angle (q) for
 * each source. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[in] ns       The number of source positions.
 * @param[in] ra       The source Right Ascensions in radians.
 * @param[in] dec      The source Declinations in radians.
 * @param[in] cos_lat  The cosine of the geographic latitude.
 * @param[in] sin_lat  The sine of the geographic latitude.
 * @param[in] lst      The local sidereal time in radians.
 * @param[out] jones   The output array of Jones matrices per source.
 */
__global__
void oskar_cudak_evaluate_jones_R_f(int ns, const float* ra,
        const float* dec, float cos_lat, float sin_lat, float lst,
        float4c* jones);

/**
 * @brief
 * CUDA kernel to construct matrices for parallactic angle rotation and
 * conversion of linear Stokes parameters from equatorial to local horizontal
 * frame (double precision).
 *
 * @details
 * This CUDA kernel constructs a set of Jones matrices that will transform the
 * equatorial linear Stokes parameters into the local horizontal frame of the
 * station. This corresponds to a rotation by the parallactic angle (q) for
 * each source. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[in] ns       The number of source positions.
 * @param[in] ra       The source Right Ascensions in radians.
 * @param[in] dec      The source Declinations in radians.
 * @param[in] cos_lat  The cosine of the geographic latitude.
 * @param[in] sin_lat  The sine of the geographic latitude.
 * @param[in] lst      The local sidereal time in radians.
 * @param[out] jones   The output array of Jones matrices per source.
 */
__global__
void oskar_cudak_evaluate_jones_R_d(int ns, const double* ra,
        const double* dec, double cos_lat, double sin_lat, double lst,
        double4c* jones);

#endif // OSKAR_CUDAK_EVALUATE_JONES_R_H_
