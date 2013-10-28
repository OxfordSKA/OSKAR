/*
 * Copyright (c) 2013, The University of Oxford
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
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
int oskar_sky_type(const oskar_Sky* sky);

OSKAR_EXPORT
int oskar_sky_location(const oskar_Sky* sky);

OSKAR_EXPORT
int oskar_sky_num_sources(const oskar_Sky* sky);

OSKAR_EXPORT
int oskar_sky_use_extended(const oskar_Sky* sky);

OSKAR_EXPORT
void oskar_sky_set_use_extended(oskar_Sky* sky, int value);

OSKAR_EXPORT
oskar_Mem* oskar_sky_ra(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_ra_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_dec(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_dec_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_I(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_I_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_Q(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_Q_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_U(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_U_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_V(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_V_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_reference_freq(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_reference_freq_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_spectral_index(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_spectral_index_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_rotation_measure(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_rotation_measure_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_l(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_l_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_m(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_m_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_n(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_n_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_radius_arcmin(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_radius_arcmin_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_fwhm_major(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_fwhm_major_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_fwhm_minor(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_fwhm_minor_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_position_angle(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_position_angle_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_a(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_a_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_b(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_b_const(const oskar_Sky* sky);

OSKAR_EXPORT
oskar_Mem* oskar_sky_gaussian_c(oskar_Sky* sky);

OSKAR_EXPORT
const oskar_Mem* oskar_sky_gaussian_c_const(const oskar_Sky* sky);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_ACCESSORS_H_ */
