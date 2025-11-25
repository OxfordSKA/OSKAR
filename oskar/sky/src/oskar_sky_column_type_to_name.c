/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


const char* oskar_sky_column_type_to_name(oskar_SkyColumn column_type)
{
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
    default:                                              /* LCOV_EXCL_LINE */
        break;                                            /* LCOV_EXCL_LINE */
    }
    return "UnknownColumn";                               /* LCOV_EXCL_LINE */
}

#ifdef __cplusplus
}
#endif
