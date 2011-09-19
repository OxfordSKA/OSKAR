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

#ifndef OSKAR_CUDA_BP2HC_H_
#define OSKAR_CUDA_BP2HC_H_

/**
 * @file oskar_cuda_bp2hc.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes a beam pattern using CUDA in the 2D horizontal coordinate system
 * (single precision).
 *
 * @details
 * Computes a beam pattern using CUDA, generating the geometric beamforming
 * weights separately. Weights can be optionally modified by an apodisation
 * function.
 *
 * The beamforming weights are NOT normalised to the number of antennas.
 *
 * This function must be supplied with the antenna x- and y-positions, the
 * test source longitude and latitude positions, the beam direction, and
 * the wavenumber.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each position of the test source.
 *
 * Apodisation is selected though the value of \p apfn which defines
 * the apodisation function applied. It takes the values:
 * -   0: No apodisation.
 * -   1: Hann Function (http://mathworld.wolfram.com/HanningFunction.html).
 *
 * Different antenna types are selected using the \p atype parameter.
 * It takes the values:
 * -   0: Isotropic antennas.
 * -   1: Antennas with sin(elevation) response.
 * -   2: Antennas with sin^2(elevation) response.
 * - 100: Antennas with Gaussian response.
 *
 * If Gaussian antennas are specified, then the normalisation and FWHM must be
 * specified in the \p gain and \p fwhm parameters.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax Array of antenna x-positions in metres.
 * @param[in] ay Array of antenna y-positions in metres.
 * @param[in] ns The number of test source positions.
 * @param[in] sa Array of test source azimuth coordinates in radians.
 * @param[in] se Array of test source elevation coordinates in radians.
 * @param[in] ba The beam azimuth direction in radians
 * @param[in] be The beam elevation direction in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[in] apfn The apodisation function type.
 * @param[in] atype The antenna type.
 * @param[in] gain The antenna multiplicative gain.
 * @param[in] fwhm For Gaussian antennas, the antenna beam FWHM in radians.
 * @param[out] image The computed beam pattern (see note, above).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
OSKAR_EXPORT
int oskar_cudaf_bp2hc(int na, const float* ax, const float* ay,
        int ns, const float* sa, const float* se, float ba, float be, float k,
        int apfn, int atype, float gain, float fwhm, float* image);

/**
 * @brief
 * Computes a beam pattern using CUDA in the 2D horizontal coordinate system
 * (double precision).
 *
 * @details
 * Computes a beam pattern using CUDA, generating the geometric beamforming
 * weights separately. Weights can be optionally modified by an apodisation
 * function.
 *
 * The beamforming weights are NOT normalised to the number of antennas.
 *
 * This function must be supplied with the antenna x- and y-positions, the
 * test source longitude and latitude positions, the beam direction, and
 * the wavenumber.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each position of the test source.
 *
 * Apodisation is selected though the value of \p apfn which defines
 * the apodisation function applied. It takes the values:
 * -   0: No apodisation.
 * -   1: Hann Function (http://mathworld.wolfram.com/HanningFunction.html).
 *
 * Different antenna types are selected using the \p atype parameter.
 * It takes the values:
 * -   0: Isotropic antennas.
 * -   1: Antennas with sin(elevation) response.
 * -   2: Antennas with sin^2(elevation) response.
 * - 100: Antennas with Gaussian response.
 *
 * If Gaussian antennas are specified, then the normalisation and FWHM must be
 * specified in the \p gain and \p fwhm parameters.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax Array of antenna x-positions in metres.
 * @param[in] ay Array of antenna y-positions in metres.
 * @param[in] ns The number of test source positions.
 * @param[in] sa Array of test source azimuth coordinates in radians.
 * @param[in] se Array of test source elevation coordinates in radians.
 * @param[in] ba The beam azimuth direction in radians
 * @param[in] be The beam elevation direction in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[in] apfn The apodisation function type.
 * @param[in] atype The antenna type.
 * @param[in] gain The antenna multiplicative gain.
 * @param[in] fwhm For Gaussian antennas, the antenna beam FWHM in radians.
 * @param[out] image The computed beam pattern (see note, above).
 *
 * @return
 * This function returns a code to indicate if there were errors in execution.
 * A return code of 0 indicates no error.
 */
OSKAR_EXPORT
int oskar_cudad_bp2hc(int na, const double* ax, const double* ay,
        int ns, const double* sa, const double* se, double ba, double be,
        double k, int apfn, int atype, double gain, double fwhm, double* image);

enum {
    apfn_none = 0,
    apfn_hann = 1
};

enum {
    atype_isotropic = 0,
    atype_sin_elevation = 1,
    atype_sin2_elevation = 2,
    atype_gaussian = 100
};

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_BP2HC_H_
