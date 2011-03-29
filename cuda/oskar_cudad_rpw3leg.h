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

#ifndef OSKAR_CUDAD_RPW3LEG_H_
#define OSKAR_CUDAD_RPW3LEG_H_

/**
 * @file oskar_cudad_rpw3leg.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes weights corresponding to relative geometric phases using CUDA.
 *
 * @details
 * Computes phase of each source relative to tracking centre.
 *
 * Returns a complex matrix of dimension (ns x na) containing
 * the complex exponential of the geometric path length difference in
 * wavenumbers relative to the phase tracking centre, for every source and
 * station.
 *
 * The output weights matrix must be pre-sized to (2 * ns * na).
 * The source index is the fastest varying dimension.
 *
 * @param[in] na Number of antennas or stations.
 * @param[in] ax Array of local equatorial antenna x-coordinates in metres.
 * @param[in] ay Array of local equatorial antenna y-coordinates in metres.
 * @param[in] az Array of local equatorial antenna z-coordinates in metres.
 * @param[in] ns Number of sources.
 * @param[in] ha Array of source Hour Angle coordinates in radians.
 * @param[in] dec Array of source Declination coordinates in radians.
 * @param[in] ha0 Hour Angle of the phase tracking centre in radians.
 * @param[in] dec0 Declination of the phase tracking centre in radians.
 * @param[in] k Wavenumber in radians / metre.
 * @param[out] weights The matrix of geometric phase weights (see note, above).
 */
void oskar_cudad_rpw3leg(int na, double* ax, double* ay, double* az, int ns,
        double* ha, double* dec, double ha0, double dec0, double k, double* weights);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDAD_RPW3LEG_H_
