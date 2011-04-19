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

#ifndef OSKAR_CUDA_LE2HG_H_
#define OSKAR_CUDA_LE2HG_H_

/**
 * @file oskar_cuda_le2hg.h
 */

#include "oskar_cuda_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transforms local equatorial to horizontal coordinates using CUDA
 * (single precision).
 *
 * @details
 * Transforms local equatorial to horizontal coordinates using CUDA.
 *
 * Coordinates can be either in device memory or in host memory (separately
 * or interleaved). The \p opt parameter must be used to specify the location:
 *
 * - If \p opt = 'd' then \p hadec is a pointer to a block of memory on the
 *   device of length (2 * ns) and contains interleaved (HA, Dec) coordinate
 *   pairs. The \p azel parameter is then also a device pointer to an array
 *   of length (2 * ns) that will contain the horizontal coordinates.
 *   No extra copying is done in this case.
 * - If \p opt = 'i' then the arrays are interleaved as above,
 *   except in host memory.
 * - If \p opt = 's' then both \p hadec and \p dec are pointers to blocks of
 *   memory on the host, each of length ns, containing separate (HA, Dec)
 *   coordinates. The \p azel and \p el parameters are then also host pointers
 *   to arrays of length ns that will contain separate horizontal coordinates.
 *
 * @param[in] opt Memory option (see notes, above).
 * @param[in] ns No. of source coordinates.
 * @param[in] hadec The right ascensions, or equatorial coordinates in radians.
 * @param[in] dec The declinations, or NULL (depending on opt).
 * @param[in] cosLat The cosine of the observer's geographic latitude.
 * @param[in] sinLat The sine of the observer's geographic latitude.
 * @param[out] azel The azimuths, or horizontal coordinates in radians.
 * @param[out] el The elevations, or NULL (depending on opt).
 */
DllExport void oskar_cudaf_le2hg(char opt, int ns, const float* hadec,
        const float* dec, float cosLat, float sinLat, float* azel, float* el);

/**
 * @brief
 * Transforms local equatorial to horizontal coordinates using CUDA
 * (double precision).
 *
 * @details
 * Transforms local equatorial to horizontal coordinates using CUDA.
 *
 * Coordinates can be either in device memory or in host memory (separately
 * or interleaved). The \p opt parameter must be used to specify the location:
 *
 * - If \p opt = 'd' then \p hadec is a pointer to a block of memory on the
 *   device of length (2 * ns) and contains interleaved (HA, Dec) coordinate
 *   pairs. The \p azel parameter is then also a device pointer to an array
 *   of length (2 * ns) that will contain the horizontal coordinates.
 *   No extra copying is done in this case.
 * - If \p opt = 'i' then the arrays are interleaved as above,
 *   except in host memory.
 * - If \p opt = 's' then both \p hadec and \p dec are pointers to blocks of
 *   memory on the host, each of length ns, containing separate (HA, Dec)
 *   coordinates. The \p azel and \p el parameters are then also host pointers
 *   to arrays of length ns that will contain separate horizontal coordinates.
 *
 * @param[in] opt Memory option (see notes, above).
 * @param[in] ns No. of source coordinates.
 * @param[in] hadec The right ascensions, or equatorial coordinates in radians.
 * @param[in] dec The declinations, or NULL (depending on opt).
 * @param[in] cosLat The cosine of the observer's geographic latitude.
 * @param[in] sinLat The sine of the observer's geographic latitude.
 * @param[out] azel The azimuths, or horizontal coordinates in radians.
 * @param[out] el The elevations, or NULL (depending on opt).
 */
DllExport void oskar_cudad_le2hg(char opt, int ns, const double* hadec,
        const double* dec, double cosLat, double sinLat, double* azel,
        double* el);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_LE2HG_H_
