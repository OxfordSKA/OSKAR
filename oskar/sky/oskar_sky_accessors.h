/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#ifndef OSKAR_SKY_ACCESSORS_H_
#define OSKAR_SKY_ACCESSORS_H_

/**
 * @file oskar_sky_accessors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the numerical precision of data in the sky model.
 *
 * @details
 * Returns the numerical precision of data in the sky model (either
 * OSKAR_SINGLE or OSKAR_DOUBLE).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
int oskar_sky_precision(const oskar_Sky* sky);

/**
 * @brief Returns the memory location of data in the sky model.
 *
 * @details
 * Returns the memory location of data in the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
int oskar_sky_mem_location(const oskar_Sky* sky);

/**
 * @brief Returns the sky model capacity.
 *
 * @details
 * Returns the sky model capacity.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
int oskar_sky_capacity(const oskar_Sky* sky);

/**
 * @brief Returns the number of sources in the sky model.
 *
 * @details
 * Returns the number of sources in the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
int oskar_sky_num_sources(const oskar_Sky* sky);

/**
 * @brief Returns the flag to specify whether the sky model contains
 * extended sources.
 *
 * @details
 * Returns the flag to specify whether the sky model contains extended sources.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
int oskar_sky_use_extended(const oskar_Sky* sky);

/**
 * @brief Sets the flag to specify whether the sky model contains
 * extended sources.
 *
 * @details
 * Sets the flag to specify whether the sky model contains extended sources.
 *
 * @param[in] sky   Pointer to sky model.
 * @param[in] value True or false.
 */
OSKAR_EXPORT
void oskar_sky_set_use_extended(oskar_Sky* sky, int value);

/**
 * @brief Returns the reference right ascension value in radians.
 *
 * @details
 * Returns the reference right ascension value in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
double oskar_sky_reference_ra_rad(const oskar_Sky* sky);

/**
 * @brief Returns the reference declination value in radians.
 *
 * @details
 * Returns the reference declination value in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
double oskar_sky_reference_dec_rad(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source right ascension values in radians.
 *
 * @details
 * Returns a handle to the source right ascension values in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_ra_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source right ascension values in radians
 * (const version).
 *
 * @details
 * Returns a handle to the source right ascension values in radians
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_ra_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source declination values in radians.
 *
 * @details
 * Returns a handle to the source declination values in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_dec_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source declination values in radians
 * (const version).
 *
 * @details
 * Returns a handle to the source declination values in radians
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_dec_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes I values in Jy.
 *
 * @details
 * Returns a handle to the source Stokes I values in Jy.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_I(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes I values in Jy
 * (const version).
 *
 * @details
 * Returns a handle to the source Stokes I values in Jy (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_I_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes Q values in Jy.
 *
 * @details
 * Returns a handle to the source Stokes Q values in Jy.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_Q(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes Q values in Jy
 * (const version).
 *
 * @details
 * Returns a handle to the source Stokes Q values in Jy (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_Q_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes U values in Jy.
 *
 * @details
 * Returns a handle to the source Stokes U values in Jy.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_U(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes U values in Jy
 * (const version).
 *
 * @details
 * Returns a handle to the source Stokes U values in Jy (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_U_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes V values in Jy.
 *
 * @details
 * Returns a handle to the source Stokes V values in Jy.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_V(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Stokes V values in Jy
 * (const version).
 *
 * @details
 * Returns a handle to the source Stokes V values in Jy (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_V_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source reference frequency values in Hz.
 *
 * @details
 * Returns a handle to the source reference frequency values in Hz.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_reference_freq_hz(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source reference frequency values in Hz
 * (const version).
 *
 * @details
 * Returns a handle to the source reference frequency values in Hz
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_reference_freq_hz_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source spectral index values.
 *
 * @details
 * Returns a handle to the source spectral index values.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_spectral_index(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source spectral index values
 * (const version).
 *
 * @details
 * Returns a handle to the source spectral index values (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_spectral_index_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source rotation measure values,
 * in radians/m^2.
 *
 * @details
 * Returns a handle to the source rotation measure values, in radians/m^2.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_rotation_measure_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source rotation measure values,
 * in radians/m^2 (const version).
 *
 * @details
 * Returns a handle to the source rotation measure values, in radians/m^2
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_rotation_measure_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source l-direction cosines.
 *
 * @details
 * Returns a handle to the source l-direction cosines,
 * relative to the reference position.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_l(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source l-direction cosines (const version).
 *
 * @details
 * Returns a handle to the source l-direction cosines,
 * relative to the reference position (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_l_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source m-direction cosines.
 *
 * @details
 * Returns a handle to the source m-direction cosines,
 * relative to the reference position.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_m(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source m-direction cosines (const version).
 *
 * @details
 * Returns a handle to the source m-direction cosines,
 * relative to the reference position (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_m_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source n-direction cosines.
 *
 * @details
 * Returns a handle to the source n-direction cosines,
 * relative to the reference position.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_n(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source n-direction cosines (const version).
 *
 * @details
 * Returns a handle to the source n-direction cosines,
 * relative to the reference position (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_n_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source FWHM major axis values in radians.
 *
 * @details
 * Returns a handle to the source FWHM major axis values in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_fwhm_major_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source FWHM major axis values in radians
 * (const version).
 *
 * @details
 * Returns a handle to the source FWHM major axis values in radians
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_fwhm_major_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source FWHM minor axis values in radians.
 *
 * @details
 * Returns a handle to the source FWHM minor axis values in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_fwhm_minor_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source FWHM minor axis values in radians
 * (const version).
 *
 * @details
 * Returns a handle to the source FWHM minor axis values in radians
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_fwhm_minor_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source major axis position angle values
 * in radians.
 *
 * @details
 * Returns a handle to the source major axis position angle values in radians.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_position_angle_rad(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source major axis position angle values
 * in radians (const version).
 *
 * @details
 * Returns a handle to the source major axis position angle values in radians
 * (const version).
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_position_angle_rad_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'a' values.
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'a' values.
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_a(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'a' values
 * (const version).
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'a' values (const version).
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_a_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'b' values.
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'b' values.
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_b(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'b' values
 * (const version).
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'b' values (const version).
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_b_const(const oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'c' values.
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'c' values.
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_c(oskar_Sky* sky);

/**
 * @brief Returns a handle to the source Gaussian parameter 'c' values
 * (const version).
 *
 * @details
 * Returns a handle to the source Gaussian parameter 'c' values (const version).
 * These values are pre-computed after loading the sky model.
 *
 * @param[in] sky Pointer to sky model.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_c_const(const oskar_Sky* sky);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_ACCESSORS_H_ */
