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

#ifndef OSKAR_CUDA_CORRELATOR_SCALAR_H_
#define OSKAR_CUDA_CORRELATOR_SCALAR_H_

/**
 * @file oskar_interferometry_cuda_correlator.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes complex visibilities (single precision).
 *
 * @details
 * Computes complex visibilities.
 *
 * Note: All pointers are device pointers, and must *not* be dereferenced
 * in host code.
 *
 * The visibilities are returned in an array of length na * (na - 1) / 2,
 * so the Hermitian conjugate is not included.
 *
 * @param[in]  num_stations  Number of antennas or stations.
 * @param[in]  station_x     Array of local equatorial station x-positions, in wavenumbers.
 * @param[in]  station_y     Array of local equatorial station y-positions, in wavenumbers.
 * @param[in]  station_z     Array of local equatorial station z-positions, in wavenumbers.

 * @param[in]  ns            Number of sources.
 * @param[in]  l             Array of source l-positions.
 * @param[in]  m             Array of source m-positions.
 * @param[in]  n             Array of source n-positions (see note, above).
 * @param[in]  eb            Matrix of E * sqrt(B) (see note, above).
 *
 * @param[in]  ra0           Right Ascension of the phase tracking centre, in radians.
 * @param[in]  dec0          Declination of the phase tracking centre, in radians.
 *
 * @param[in]  lst0          The local sidereal time at the start of the correlator dump.
 * @param[in]  nsdt          The number of averaging cycles to do.
 * @param[in]  sdt           The time interval between averages in seconds.
 * @param[in]  lambda_bandwidth  Wavelength (m) times bandwidth (Hz).
 * @param[in]  work_k        Work array for phase matrix (ns * na).
 * @param[in]  work_uvw      Work array for uvw coordinates (3*na)
 * @param[out] vis           Array of visibilities (see note, above).
 */
OSKAR_EXPORT
int oskar_cuda_correlator_scalar_f(int num_stations, const float* station_x,
        const float* station_y, const float* station_z,
        int ns, const float* l,
        const float* m, const float* n, const float* b, const float2* e,
        float ra0, float dec0, float lst0, int nsdt, float sdt,
        float lambda_bandwidth, float2* work_k, float* work_uvw, float2* vis);




/**
 * @brief
 * Computes complex visibilities (double precision).
 *
 * @details
 * Computes complex visibilities.
 *
 * Note that all pointers are device pointers, and must *not* be dereferenced
 * in host code.
 *
 * The visibilities and their (u,v,w) coordinates are returned in arrays of
 * length na * (na - 1) / 2, so the Hermitian conjugate is not included.
 *
 * @param[in]  num_stations  Number of antennas or stations.
 * @param[in]  station_x     Array of local equatorial station x-positions, in wavenumbers.
 * @param[in]  station_y     Array of local equatorial station y-positions, in wavenumbers.
 * @param[in]  station_z     Array of local equatorial station z-positions, in wavenumbers.

 * @param[in]  ns        Number of sources.
 * @param[in]  l         Array of source l-positions.
 * @param[in]  m         Array of source m-positions.
 * @param[in]  n         Array of source n-positions (see note, above).
 *
 * @param[in]  eb        Matrix of E * sqrt(B) (see note, above).
 *
 * @param[in]  ra0       Right Ascension of the phase tracking centre, in radians.
 * @param[in]  dec0      Declination of the phase tracking centre, in radians.
 *
 * @param[in]  lst0      The local sidereal time at the start of the correlator dump.
 * @param[in]  nsdt      The number of averaging cycles to do.
 * @param[in]  sdt       The time interval between averages in seconds.
 * @param[in]  lambda_bandwidth Wavelength (m) times bandwidth (Hz).
 * @param[in]  work_k    Work array for phase matrix (ns * na).
 * @param[in]  work_uvw  Work array for uvw coordinates (3*na)
 * @param[out] vis       Array of visibilities (see note, above).
 */
OSKAR_EXPORT
int oskar_cuda_correlator_scalar_d(int na, const double* ax,
        const double* ay, const double* az, int ns, const double* l,
        const double* m, const double* n, const double* b, const double2* e,
        double ra0, double dec0, double lst0, int nsdt, double sdt,
        double lambda_bandwidth, double2* work_k, double* work_uvw, double2* vis);
#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_CORRELATOR_SCALAR_H_
