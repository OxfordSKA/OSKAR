/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Case-insensitive string comparison. */
static int matches(const char* str_a, const char* str_b)
{
    while (*str_a && *str_b)
    {
        if (tolower(*str_a++) != tolower(*str_b++)) return 0;
    }
    return (*str_a == '\0' && *str_b == '\0');
}


/* Case-insensitive check if string starts with prefix. */
static int starts_with(const char* str, const char* prefix)
{
    while (*prefix && *str)
    {
        if (tolower(*str) != tolower(*prefix)) return 0;
        str++;
        prefix++;
    }
    return (*prefix == '\0');
}


oskar_SkyColumn oskar_sky_column_type_from_name(const char* name)
{
    if (starts_with(name, "RaD") ||
            starts_with(name, "ra_deg"))
    {
        return OSKAR_SKY_RA_DEG;
    }
    if (starts_with(name, "DecD") ||
            starts_with(name, "dec_deg"))
    {
        return OSKAR_SKY_DEC_DEG;
    }
    if (starts_with(name, "RA"))
    {
        return OSKAR_SKY_RA_RAD;
    }
    if (starts_with(name, "DE"))
    {
        return OSKAR_SKY_DEC_RAD;
    }
    if (matches(name, "I") ||
            matches(name, "StokesI") ||
            starts_with(name, "i_pol") ||
            starts_with(name, "STOKES_I"))
    {
        return OSKAR_SKY_I_JY;
    }
    if (matches(name, "Q") ||
            matches(name, "StokesQ") ||
            starts_with(name, "q_pol") ||
            starts_with(name, "STOKES_Q"))
    {
        return OSKAR_SKY_Q_JY;
    }
    if (matches(name, "U") ||
            matches(name, "StokesU") ||
            starts_with(name, "u_pol") ||
            starts_with(name, "STOKES_U"))
    {
        return OSKAR_SKY_U_JY;
    }
    if (matches(name, "V") ||
            matches(name, "StokesV") ||
            starts_with(name, "v_pol") ||
            starts_with(name, "STOKES_V"))
    {
        return OSKAR_SKY_V_JY;
    }
    if (starts_with(name, "ReferenceF") ||
            starts_with(name, "RefFreq") ||
            starts_with(name, "ref_freq"))
    {
        return OSKAR_SKY_REF_HZ;
    }
    if (matches(name, "SpectralIndex") ||
            matches(name, "SpInx") ||
            matches(name, "SpIdx") ||
            matches(name, "SPIX") ||
            matches(name, "spec_idx") ||
            starts_with(name, "alpha"))
    {
        return OSKAR_SKY_SPEC_IDX;
    }
    if (starts_with(name, "Rotat") ||
            matches(name, "rot_meas"))
    {
        return OSKAR_SKY_RM_RAD;
    }
    if (starts_with(name, "Maj"))
    {
        return OSKAR_SKY_MAJOR_RAD;
    }
    if (starts_with(name, "Min"))
    {
        return OSKAR_SKY_MINOR_RAD;
    }
    if (starts_with(name, "Ori") ||
            matches(name, "PositionAngle") ||
            starts_with(name, "pos_ang"))
    {
        return OSKAR_SKY_PA_RAD;
    }
    if (starts_with(name, "SpectralCurv"))
    {
        return OSKAR_SKY_SPEC_CURV;
    }
    if (matches(name, "LogarithmicSI") ||
            matches(name, "log_spec_idx"))
    {
        return OSKAR_SKY_LIN_SI;
    }
    if (starts_with(name, "PolarizationAngle") ||
            starts_with(name, "PolarisationAngle") ||
            starts_with(name, "pol_ang") ||
            matches(name, "POLA"))
    {
        return OSKAR_SKY_POLA_RAD;
    }
    if (matches(name, "PolarizedFraction") ||
            matches(name, "PolarisedFraction") ||
            starts_with(name, "pol_frac") ||
            matches(name, "POLF"))
    {
        return OSKAR_SKY_POLF;
    }
    if (starts_with(name, "ReferenceW") ||
            starts_with(name, "RefWave"))
    {
        return OSKAR_SKY_REF_WAVE_M;
    }
    if (starts_with(name, "LineWidth"))
    {
        return OSKAR_SKY_LINE_WIDTH_HZ;
    }

    /* Unknown column type - ignore on load. */
    return OSKAR_SKY_CUSTOM;
}

#ifdef __cplusplus
}
#endif
