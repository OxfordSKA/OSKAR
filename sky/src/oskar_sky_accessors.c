/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <private_sky.h>
#include <oskar_sky_accessors.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_precision(const oskar_Sky* sky)
{
    return sky->precision;
}

int oskar_sky_mem_location(const oskar_Sky* sky)
{
    return sky->mem_location;
}

int oskar_sky_capacity(const oskar_Sky* sky)
{
    return sky->capacity;
}

int oskar_sky_num_sources(const oskar_Sky* sky)
{
    return sky->num_sources;
}

int oskar_sky_use_extended(const oskar_Sky* sky)
{
    return sky->use_extended;
}

void oskar_sky_set_use_extended(oskar_Sky* sky, int value)
{
    sky->use_extended = value;
}

double oskar_sky_reference_ra_rad(const oskar_Sky* sky)
{
    return sky->reference_ra_rad;
}

double oskar_sky_reference_dec_rad(const oskar_Sky* sky)
{
    return sky->reference_dec_rad;
}

oskar_Mem* oskar_sky_ra_rad(oskar_Sky* sky)
{
    return sky->ra_rad;
}

const oskar_Mem* oskar_sky_ra_rad_const(const oskar_Sky* sky)
{
    return sky->ra_rad;
}

oskar_Mem* oskar_sky_dec_rad(oskar_Sky* sky)
{
    return sky->dec_rad;
}

const oskar_Mem* oskar_sky_dec_rad_const(const oskar_Sky* sky)
{
    return sky->dec_rad;
}

oskar_Mem* oskar_sky_I(oskar_Sky* sky)
{
    return sky->I;
}

const oskar_Mem* oskar_sky_I_const(const oskar_Sky* sky)
{
    return sky->I;
}

oskar_Mem* oskar_sky_Q(oskar_Sky* sky)
{
    return sky->Q;
}

const oskar_Mem* oskar_sky_Q_const(const oskar_Sky* sky)
{
    return sky->Q;
}

oskar_Mem* oskar_sky_U(oskar_Sky* sky)
{
    return sky->U;
}

const oskar_Mem* oskar_sky_U_const(const oskar_Sky* sky)
{
    return sky->U;
}

oskar_Mem* oskar_sky_V(oskar_Sky* sky)
{
    return sky->V;
}

const oskar_Mem* oskar_sky_V_const(const oskar_Sky* sky)
{
    return sky->V;
}

oskar_Mem* oskar_sky_reference_freq_hz(oskar_Sky* sky)
{
    return sky->reference_freq_hz;
}

const oskar_Mem* oskar_sky_reference_freq_hz_const(const oskar_Sky* sky)
{
    return sky->reference_freq_hz;
}

oskar_Mem* oskar_sky_spectral_index(oskar_Sky* sky)
{
    return sky->spectral_index;
}

const oskar_Mem* oskar_sky_spectral_index_const(const oskar_Sky* sky)
{
    return sky->spectral_index;
}

oskar_Mem* oskar_sky_rotation_measure_rad(oskar_Sky* sky)
{
    return sky->rm_rad;
}

const oskar_Mem* oskar_sky_rotation_measure_rad_const(const oskar_Sky* sky)
{
    return sky->rm_rad;
}

oskar_Mem* oskar_sky_l(oskar_Sky* sky)
{
    return sky->l;
}

const oskar_Mem* oskar_sky_l_const(const oskar_Sky* sky)
{
    return sky->l;
}

oskar_Mem* oskar_sky_m(oskar_Sky* sky)
{
    return sky->m;
}

const oskar_Mem* oskar_sky_m_const(const oskar_Sky* sky)
{
    return sky->m;
}

oskar_Mem* oskar_sky_n(oskar_Sky* sky)
{
    return sky->n;
}

const oskar_Mem* oskar_sky_n_const(const oskar_Sky* sky)
{
    return sky->n;
}

oskar_Mem* oskar_sky_fwhm_major_rad(oskar_Sky* sky)
{
    return sky->fwhm_major_rad;
}

const oskar_Mem* oskar_sky_fwhm_major_rad_const(const oskar_Sky* sky)
{
    return sky->fwhm_major_rad;
}

oskar_Mem* oskar_sky_fwhm_minor_rad(oskar_Sky* sky)
{
    return sky->fwhm_minor_rad;
}

const oskar_Mem* oskar_sky_fwhm_minor_rad_const(const oskar_Sky* sky)
{
    return sky->fwhm_minor_rad;
}

oskar_Mem* oskar_sky_position_angle_rad(oskar_Sky* sky)
{
    return sky->pa_rad;
}

const oskar_Mem* oskar_sky_position_angle_rad_const(const oskar_Sky* sky)
{
    return sky->pa_rad;
}

oskar_Mem* oskar_sky_gaussian_a(oskar_Sky* sky)
{
    return sky->gaussian_a;
}

const oskar_Mem* oskar_sky_gaussian_a_const(const oskar_Sky* sky)
{
    return sky->gaussian_a;
}

oskar_Mem* oskar_sky_gaussian_b(oskar_Sky* sky)
{
    return sky->gaussian_b;
}

const oskar_Mem* oskar_sky_gaussian_b_const(const oskar_Sky* sky)
{
    return sky->gaussian_b;
}

oskar_Mem* oskar_sky_gaussian_c(oskar_Sky* sky)
{
    return sky->gaussian_c;
}

const oskar_Mem* oskar_sky_gaussian_c_const(const oskar_Sky* sky)
{
    return sky->gaussian_c;
}

#if 0
int oskar_sky_num_filter_bands(const oskar_Sky* sky)
{
    return sky->num_filter_bands;
}

oskar_Mem* oskar_sky_filter_band_radius_rad(oskar_Sky* sky)
{
    return sky->filter_band_radius_rad;
}

const oskar_Mem* oskar_sky_filter_band_radius_rad_const(const oskar_Sky* sky)
{
    return sky->filter_band_radius_rad;
}

oskar_Mem* oskar_sky_filter_band_flux_jy(oskar_Sky* sky)
{
    return sky->filter_band_flux_jy;
}

const oskar_Mem* oskar_sky_filter_band_flux_jy_const(const oskar_Sky* sky)
{
    return sky->filter_band_flux_jy;
}
#endif

#ifdef __cplusplus
}
#endif
