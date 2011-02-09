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

#include "cuda/_equatorialToHorizontalGeneric.h"

/**
 * @details
 * This CUDA kernel transforms sources specified in a generic equatorial
 * system (RA, Dec) to local horizontal coordinates (azimuth, elevation).
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
void _equatorialToHorizontalGeneric(const int ns, const float2* radec,
        const float cosLat, const float sinLat, const float lst, float2* azel)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    // Copy source coordinates from global memory.
    float2 src = radec[s];

    // Precompute.
    float cosDec, sinDec, cosHA, sinHA;
    const float hourAngle = lst - src.x; // LST - RA
    sincosf(src.y, &sinDec, &cosDec);
    sincosf(hourAngle, &sinHA, &cosHA);
    const float f = cosDec * cosHA;

    // Find azimuth and elevation.
    const float Y1 = -cosDec * sinHA;
    const float X1 = cosLat * sinDec - sinLat * f;
    const float Y2 = sinLat * sinDec + cosLat * f;
    const float X2 = hypotf(X1, Y1); // sqrtf(X1*X1 + Y1*Y1);
    azel[s].x = atan2f(Y1, X1);
    azel[s].y = atan2f(Y2, X2);
}
