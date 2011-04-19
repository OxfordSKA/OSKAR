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

#ifndef OSKAR_CUDAK_EQ2HG_H_
#define OSKAR_CUDAK_EQ2HG_H_

/**
 * @file oskar_cudak_eq2hg.h
 */

#include "cuda/CudaEclipse.h"

/**
 * @brief
 * CUDA kernel to compute local source positions (single precision).
 *
 * @details
 * This CUDA kernel transforms sources specified in a generic equatorial
 * system (RA, Dec) to local horizontal coordinates (azimuth, elevation).
 *
 * The kernel requires (threads per block) * sizeof(float2) bytes of
 * shared memory.
 *
 * Each thread operates on a single source. The source positions are
 * specified as (RA, Dec) pairs in the \p radec array:
 *
 * radec.x = {RA}
 * radec.y = {Dec}
 *
 * The output \p azel array contains the azimuth and elevation pairs for each
 * source:
 *
 * azel.x = {azimuth}
 * azel.y = {elevation}
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: 4 * ns.
 * \li Arctangents: 2 * ns.
 * \li Multiplies: 8 * ns.
 * \li Additions / subtractions: 4 * ns.
 * \li Square roots: ns.
 *
 * @param[in] ns The number of source positions.
 * @param[in] radec The RA and Declination source coordinates in radians.
 * @param[in] cosLat The cosine of the geographic latitude.
 * @param[in] sinLat The sine of the geographic latitude.
 * @param[in] lst The Local Sidereal Time (= ST + geographic longitude) in radians.
 * @param[out] azel The azimuth and elevation source coordinates in radians.
 */
__global__
void oskar_cudakf_eq2hg(int ns, const float2* radec,
        float cosLat, float sinLat, float lst, float2* azel);

/**
 * @brief
 * CUDA kernel to compute local source positions (double precision).
 *
 * @details
 * This CUDA kernel transforms sources specified in a generic equatorial
 * system (RA, Dec) to local horizontal coordinates (azimuth, elevation).
 *
 * The kernel requires (threads per block) * sizeof(double2) bytes of
 * shared memory.
 *
 * Each thread operates on a single source. The source positions are
 * specified as (RA, Dec) pairs in the \p radec array:
 *
 * radec.x = {RA}
 * radec.y = {Dec}
 *
 * The output \p azel array contains the azimuth and elevation pairs for each
 * source:
 *
 * azel.x = {azimuth}
 * azel.y = {elevation}
 *
 * The number of doubleing-point operations performed by this kernel is:
 * \li Sines and cosines: 4 * ns.
 * \li Arctangents: 2 * ns.
 * \li Multiplies: 8 * ns.
 * \li Additions / subtractions: 4 * ns.
 * \li Square roots: ns.
 *
 * @param[in] ns The number of source positions.
 * @param[in] radec The RA and Declination source coordinates in radians.
 * @param[in] cosLat The cosine of the geographic latitude.
 * @param[in] sinLat The sine of the geographic latitude.
 * @param[in] lst The Local Sidereal Time (= ST + geographic longitude) in radians.
 * @param[out] azel The azimuth and elevation source coordinates in radians.
 */
__global__
void oskar_cudakd_eq2hg(int ns, const double2* radec,
        double cosLat, double sinLat, double lst, double2* azel);

#endif // OSKAR_CUDAK_EQ2HG_H_
