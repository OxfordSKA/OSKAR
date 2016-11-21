/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_SKY_SET_SOURCE_H_
#define OSKAR_SKY_SET_SOURCE_H_

/**
 * @file oskar_sky_set_source.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets source data into a sky model.
 *
 * @details
 * This function sets sky model data for a single source at the given index.
 * The sky model must already be large enough to hold the source data.
 *
 * @param[in,out] sky            Pointer to sky model.
 * @param[in] index              Source index in sky model to set.
 * @param[in] ra_rad             Source right ascension in radians.
 * @param[in] dec_rad            Source declination in radians.
 * @param[in] I                  Source Stokes I in Jy.
 * @param[in] Q                  Source Stokes Q in Jy.
 * @param[in] U                  Source Stokes U in Jy.
 * @param[in] V                  Source Stokes V in Jy.
 * @param[in] ref_frequency_hz   Source reference frequency in Hz.
 * @param[in] spectral_index     Source spectral index.
 * @param[in] rotation_measure   Source rotation measure in radians/m^2.
 * @param[in] fwhm_major_rad     Gaussian source major axis FWHM, in radians.
 * @param[in] fwhm_minor_rad     Gaussian source minor axis FWHM, in radians.
 * @param[in] position_angle_rad Gaussian source position angle, in radians.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_sky_set_source(oskar_Sky* sky, int index, double ra_rad,
        double dec_rad, double I, double Q, double U, double V,
        double ref_frequency_hz, double spectral_index, double rotation_measure,
        double fwhm_major_rad, double fwhm_minor_rad, double position_angle_rad,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_SET_SOURCE_H_ */
