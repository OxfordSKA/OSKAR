/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky_accessors.h"

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

/* NOLINTNEXTLINE(readability-identifier-naming) */
oskar_Mem* oskar_sky_I(oskar_Sky* sky)
{
    return sky->I;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
const oskar_Mem* oskar_sky_I_const(const oskar_Sky* sky)
{
    return sky->I;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
oskar_Mem* oskar_sky_Q(oskar_Sky* sky)
{
    return sky->Q;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
const oskar_Mem* oskar_sky_Q_const(const oskar_Sky* sky)
{
    return sky->Q;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
oskar_Mem* oskar_sky_U(oskar_Sky* sky)
{
    return sky->U;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
const oskar_Mem* oskar_sky_U_const(const oskar_Sky* sky)
{
    return sky->U;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
oskar_Mem* oskar_sky_V(oskar_Sky* sky)
{
    return sky->V;
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
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

#ifdef __cplusplus
}
#endif
