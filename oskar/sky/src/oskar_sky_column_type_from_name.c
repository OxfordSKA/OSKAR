/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ctype.h>
#include <stdlib.h>

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
        if (tolower(*str++) != tolower(*prefix++)) return 0;
    }
    return (*prefix == '\0');
}


oskar_SkyColumn oskar_sky_column_type_from_name(
        const char* name,
        double* suffix
)
{
    /* Check for a numeric suffix first. */
    if (suffix)
    {
        const char* p = name;
        *suffix = 0.0;
        for (; *p; ++p)
        {
            char* end = 0;
            if (!(isdigit(*p) || *p == '+' || *p == '-' || *p == '.'))
            {
                continue;
            }
            const double value = strtod(p, &end);
            if (end == p) continue;
            if (*end == '\0')
            {
                *suffix = value;
                break;
            }
        }
    }

    /* Check against unambiguous known names. */
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
            starts_with(name, "I_") ||
            starts_with(name, "StokesI") ||
            starts_with(name, "STOKES_I") ||
            starts_with(name, "int_flux") ||
            starts_with(name, "Fint"))
    {
        return OSKAR_SKY_I_JY;
    }
    if (matches(name, "Q") ||
            starts_with(name, "Q_") ||
            starts_with(name, "StokesQ") ||
            starts_with(name, "STOKES_Q"))
    {
        return OSKAR_SKY_Q_JY;
    }
    if (matches(name, "U") ||
            starts_with(name, "U_") ||
            starts_with(name, "StokesU") ||
            starts_with(name, "STOKES_U"))
    {
        return OSKAR_SKY_U_JY;
    }
    if (matches(name, "V") ||
            starts_with(name, "V_") ||
            starts_with(name, "StokesV") ||
            starts_with(name, "STOKES_V"))
    {
        return OSKAR_SKY_V_JY;
    }
    if (starts_with(name, "ReferenceFreq") ||
            starts_with(name, "RefFreq") ||
            starts_with(name, "reference_freq") ||
            starts_with(name, "ref_freq"))
    {
        return OSKAR_SKY_REF_HZ;
    }
    if (starts_with(name, "FrequencyInc") ||
            starts_with(name, "FreqInc") ||
            starts_with(name, "frequency_inc") ||
            starts_with(name, "freq_inc"))
    {
        return OSKAR_SKY_INC_HZ;
    }
    if (starts_with(name, "SpectralCurv") ||
            starts_with(name, "spectral_curv") ||
            starts_with(name, "spec_curv"))
    {
        return OSKAR_SKY_SPEC_CURV;
    }
    if (starts_with(name, "Sp") || /* Check after spectral curvature. */
            starts_with(name, "alpha"))
    {
        return OSKAR_SKY_SPEC_IDX;
    }
    if (starts_with(name, "Rot") ||
            starts_with(name, "RM"))
    {
        return OSKAR_SKY_RM_RAD;
    }
    if (matches(name, "LogarithmicSI") ||
            matches(name, "log_spec_idx"))
    {
        return OSKAR_SKY_LIN_SI;
    }
    if (starts_with(name, "PolarizedFrac") ||
            starts_with(name, "PolarisedFrac") ||
            starts_with(name, "polarised_frac") ||
            starts_with(name, "polarised_frac") ||
            starts_with(name, "pol_frac") ||
            matches(name, "POLF"))
    {
        return OSKAR_SKY_POLF;
    }
    if (starts_with(name, "PolarizationAng") ||
            starts_with(name, "PolarisationAng") ||
            starts_with(name, "polarization_ang") ||
            starts_with(name, "polarisation_ang") ||
            starts_with(name, "pol_ang") ||
            matches(name, "POLA"))
    {
        return OSKAR_SKY_POLA_RAD;
    }
    if (starts_with(name, "ReferenceWave") ||
            starts_with(name, "RefWave") ||
            starts_with(name, "reference_wave") ||
            starts_with(name, "ref_wave"))
    {
        return OSKAR_SKY_REF_WAVE_M;
    }
    if (starts_with(name, "LineWidth") ||
            starts_with(name, "line_width"))
    {
        return OSKAR_SKY_LINE_WIDTH_HZ;
    }
    if (starts_with(name, "Maj"))
    {
        return OSKAR_SKY_MAJOR_RAD;
    }
    if (starts_with(name, "Min"))
    {
        return OSKAR_SKY_MINOR_RAD;
    }
    if (starts_with(name, "SemiMaj") ||
            starts_with(name, "semi_maj") ||
            starts_with(name, "a"))
    {
        return OSKAR_SKY_SEMI_MAJOR;
    }
    if (starts_with(name, "SemiMin") ||
            starts_with(name, "semi_min") ||
            starts_with(name, "b"))
    {
        return OSKAR_SKY_SEMI_MINOR;
    }
    if (starts_with(name, "Ori") ||
            starts_with(name, "PositionAng") ||
            starts_with(name, "position_ang") ||
            starts_with(name, "pos_ang") ||
            starts_with(name, "pa"))
    {
        return OSKAR_SKY_PA_RAD;
    }

    /* Unknown column type - ignore on load. */
    return OSKAR_SKY_CUSTOM;
}

#ifdef __cplusplus
}
#endif
