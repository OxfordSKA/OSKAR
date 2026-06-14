/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


const char* oskar_sky_column_type_to_name(
        oskar_SkyColumn column_type,
        int use_ska_convention
)
{
    if (use_ska_convention)
    {
        switch (column_type)
        {
        case OSKAR_SKY_RA_RAD: /* Unused */               /* LCOV_EXCL_LINE */
            return "ra_rad";                              /* LCOV_EXCL_LINE */
        case OSKAR_SKY_DEC_RAD: /* Unused */              /* LCOV_EXCL_LINE */
            return "dec_rad";                             /* LCOV_EXCL_LINE */
        case OSKAR_SKY_RA_DEG:
            return "ra_deg";
        case OSKAR_SKY_DEC_DEG:
            return "dec_deg";
        case OSKAR_SKY_I_JY:
            return "i_pol_jy";
        case OSKAR_SKY_Q_JY:
            return "q_pol_jy";
        case OSKAR_SKY_U_JY:
            return "u_pol_jy";
        case OSKAR_SKY_V_JY:
            return "v_pol_jy";
        case OSKAR_SKY_REF_HZ:
            return "ref_freq_hz";
        case OSKAR_SKY_SPEC_IDX:
            return "spec_idx";
        case OSKAR_SKY_RM_RAD:
            return "rot_meas";
        case OSKAR_SKY_MAJOR_RAD: /* Unused */            /* LCOV_EXCL_LINE */
            return "major_ax_arcsec";                     /* LCOV_EXCL_LINE */
        case OSKAR_SKY_MINOR_RAD: /* Unused */            /* LCOV_EXCL_LINE */
            return "minor_ax_arcsec";                     /* LCOV_EXCL_LINE */
        case OSKAR_SKY_SEMI_MAJOR:
            return "a_arcsec";
        case OSKAR_SKY_SEMI_MINOR:
            return "b_arcsec";
        case OSKAR_SKY_PA_RAD:
            return "pa_deg";
        case OSKAR_SKY_LIN_SI:
            return "log_spec_idx";
        case OSKAR_SKY_POLA_RAD:
            return "pol_ang_deg";
        case OSKAR_SKY_POLF:
            return "pol_frac";
        case OSKAR_SKY_REF_WAVE_M:
            return "ref_wavelength_m";
        case OSKAR_SKY_SPEC_CURV:
            return "spec_curv";
        case OSKAR_SKY_LINE_WIDTH_HZ:
            return "line_width_hz";
        default:                                          /* LCOV_EXCL_LINE */
            break;                                        /* LCOV_EXCL_LINE */
        }
        return "unknown";                                 /* LCOV_EXCL_LINE */
    }
    switch (column_type)
    {
    case OSKAR_SKY_RA_RAD:
        return "Ra";
    case OSKAR_SKY_RA_DEG:
        return "RaD";
    case OSKAR_SKY_DEC_RAD:
        return "Dec";
    case OSKAR_SKY_DEC_DEG:
        return "DecD";
    case OSKAR_SKY_I_JY:
        return "I";
    case OSKAR_SKY_Q_JY:
        return "Q";
    case OSKAR_SKY_U_JY:
        return "U";
    case OSKAR_SKY_V_JY:
        return "V";
    case OSKAR_SKY_REF_HZ:
        return "ReferenceFrequency";
    case OSKAR_SKY_SPEC_IDX:
        return "SpectralIndex";
    case OSKAR_SKY_RM_RAD:
        return "RotationMeasure";
    case OSKAR_SKY_MAJOR_RAD:
        return "MajorAxis";
    case OSKAR_SKY_MINOR_RAD:
        return "MinorAxis";
    case OSKAR_SKY_SEMI_MAJOR:
        return "SemiMajorAxis";
    case OSKAR_SKY_SEMI_MINOR:
        return "SemiMinorAxis";
    case OSKAR_SKY_PA_RAD:
        return "Orientation";
    case OSKAR_SKY_LIN_SI:
        return "LogarithmicSI";
    case OSKAR_SKY_POLA_RAD:
        return "PolarizationAngle";
    case OSKAR_SKY_POLF:
        return "PolarizedFraction";
    case OSKAR_SKY_REF_WAVE_M:
        return "ReferenceWavelength";
    case OSKAR_SKY_SPEC_CURV:
        return "SpectralCurvature";
    case OSKAR_SKY_LINE_WIDTH_HZ:
        return "LineWidth";
    default:                                              /* LCOV_EXCL_LINE */
        break;                                            /* LCOV_EXCL_LINE */
    }
    return "UnknownColumn";                               /* LCOV_EXCL_LINE */
}

#ifdef __cplusplus
}
#endif
