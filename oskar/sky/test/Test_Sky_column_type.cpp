/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"


TEST(Sky, column_type_from_name)
{
    double suffix = 0.0;
    ASSERT_EQ(OSKAR_SKY_RA_RAD,
            oskar_sky_column_type_from_name("Ra", 0)
    );
    ASSERT_EQ(OSKAR_SKY_DEC_RAD,
            oskar_sky_column_type_from_name("Dec", 0)
    );
    ASSERT_EQ(OSKAR_SKY_RA_DEG,
            oskar_sky_column_type_from_name("RaD", 0)
    );
    ASSERT_EQ(OSKAR_SKY_DEC_DEG,
            oskar_sky_column_type_from_name("DecD", 0)
    );
    ASSERT_EQ(OSKAR_SKY_RA_DEG,
            oskar_sky_column_type_from_name("ra_deg", 0)
    );
    ASSERT_EQ(OSKAR_SKY_DEC_DEG,
            oskar_sky_column_type_from_name("dec_deg", 0)
    );
    ASSERT_EQ(OSKAR_SKY_I_JY,
            oskar_sky_column_type_from_name("I", 0)
    );
    ASSERT_EQ(OSKAR_SKY_I_JY,
            oskar_sky_column_type_from_name("i_pol_jy", 0)
    );
    ASSERT_EQ(OSKAR_SKY_I_JY,
            oskar_sky_column_type_from_name("StokesI", 0)
    );
    ASSERT_EQ(OSKAR_SKY_I_JY,
            oskar_sky_column_type_from_name("Fint143", &suffix)
    );
    ASSERT_DOUBLE_EQ(143.0, suffix);
    ASSERT_EQ(OSKAR_SKY_Q_JY,
            oskar_sky_column_type_from_name("Q", 0)
    );
    ASSERT_EQ(OSKAR_SKY_Q_JY,
            oskar_sky_column_type_from_name("StokesQ", &suffix)
    );
    ASSERT_DOUBLE_EQ(0.0, suffix);
    ASSERT_EQ(OSKAR_SKY_U_JY,
            oskar_sky_column_type_from_name("U", 0)
    );
    ASSERT_EQ(OSKAR_SKY_U_JY,
            oskar_sky_column_type_from_name("StokesU", 0)
    );
    ASSERT_EQ(OSKAR_SKY_V_JY,
            oskar_sky_column_type_from_name("V", 0)
    );
    ASSERT_EQ(OSKAR_SKY_V_JY,
            oskar_sky_column_type_from_name("StokesV", 0)
    );
    ASSERT_EQ(OSKAR_SKY_REF_HZ,
            oskar_sky_column_type_from_name("ReferenceFrequency", 0)
    );
    ASSERT_EQ(OSKAR_SKY_REF_HZ,
            oskar_sky_column_type_from_name("ref_freq_hz", 0)
    );
    ASSERT_EQ(OSKAR_SKY_SPEC_IDX,
            oskar_sky_column_type_from_name("SpectralIndex", 0)
    );
    ASSERT_EQ(OSKAR_SKY_SPEC_IDX,
            oskar_sky_column_type_from_name("spec_idx", 0)
    );
    ASSERT_EQ(OSKAR_SKY_RM_RAD,
            oskar_sky_column_type_from_name("RotationMeasure", 0)
    );
    ASSERT_EQ(OSKAR_SKY_MAJOR_RAD,
            oskar_sky_column_type_from_name("MajorAxis", 0)
    );
    ASSERT_EQ(OSKAR_SKY_MINOR_RAD,
            oskar_sky_column_type_from_name("MinorAxis", 0)
    );
    ASSERT_EQ(OSKAR_SKY_SEMI_MAJOR,
            oskar_sky_column_type_from_name("a_arcsec", 0)
    );
    ASSERT_EQ(OSKAR_SKY_SEMI_MINOR,
            oskar_sky_column_type_from_name("b_arcsec", 0)
    );
    ASSERT_EQ(OSKAR_SKY_PA_RAD,
            oskar_sky_column_type_from_name("Orientation", 0)
    );
    ASSERT_EQ(OSKAR_SKY_PA_RAD,
            oskar_sky_column_type_from_name("pa_deg", 0)
    );
    ASSERT_EQ(OSKAR_SKY_POLA_RAD,
            oskar_sky_column_type_from_name("PolarizationAngle", 0)
    );
    ASSERT_EQ(OSKAR_SKY_POLF,
            oskar_sky_column_type_from_name("PolarizedFraction", 0)
    );
    ASSERT_EQ(OSKAR_SKY_REF_WAVE_M,
            oskar_sky_column_type_from_name("ReferenceWavelength", 0)
    );
    ASSERT_EQ(OSKAR_SKY_SPEC_CURV,
            oskar_sky_column_type_from_name("SpectralCurvature", 0)
    );
    ASSERT_EQ(OSKAR_SKY_LINE_WIDTH_HZ,
            oskar_sky_column_type_from_name("LineWidth", 0)
    );
    ASSERT_EQ(OSKAR_SKY_LINE_WIDTH_HZ,
            oskar_sky_column_type_from_name("line_width_hz", 0)
    );
}


TEST(Sky, column_type_to_name)
{
    ASSERT_STREQ("Ra",
            oskar_sky_column_type_to_name(OSKAR_SKY_RA_RAD, 0)
    );
    ASSERT_STREQ("Dec",
            oskar_sky_column_type_to_name(OSKAR_SKY_DEC_RAD, 0)
    );
    ASSERT_STREQ("RaD",
            oskar_sky_column_type_to_name(OSKAR_SKY_RA_DEG, 0)
    );
    ASSERT_STREQ("DecD",
            oskar_sky_column_type_to_name(OSKAR_SKY_DEC_DEG, 0)
    );
    ASSERT_STREQ("ra_deg",
            oskar_sky_column_type_to_name(OSKAR_SKY_RA_DEG, 1)
    );
    ASSERT_STREQ("dec_deg",
            oskar_sky_column_type_to_name(OSKAR_SKY_DEC_DEG, 1)
    );
    ASSERT_STREQ("I",
            oskar_sky_column_type_to_name(OSKAR_SKY_I_JY, 0)
    );
    ASSERT_STREQ("i_pol_jy",
            oskar_sky_column_type_to_name(OSKAR_SKY_I_JY, 1)
    );
    ASSERT_STREQ("Q",
            oskar_sky_column_type_to_name(OSKAR_SKY_Q_JY, 0)
    );
    ASSERT_STREQ("U",
            oskar_sky_column_type_to_name(OSKAR_SKY_U_JY, 0)
    );
    ASSERT_STREQ("V",
            oskar_sky_column_type_to_name(OSKAR_SKY_V_JY, 0)
    );
    ASSERT_STREQ(
            "ReferenceFrequency",
            oskar_sky_column_type_to_name(OSKAR_SKY_REF_HZ, 0)
    );
    ASSERT_STREQ(
            "ref_freq_hz",
            oskar_sky_column_type_to_name(OSKAR_SKY_REF_HZ, 1)
    );
    ASSERT_STREQ(
            "SpectralIndex",
            oskar_sky_column_type_to_name(OSKAR_SKY_SPEC_IDX, 0)
    );
    ASSERT_STREQ(
            "spec_idx",
            oskar_sky_column_type_to_name(OSKAR_SKY_SPEC_IDX, 1)
    );
    ASSERT_STREQ(
            "RotationMeasure",
            oskar_sky_column_type_to_name(OSKAR_SKY_RM_RAD, 0)
    );
    ASSERT_STREQ(
            "MajorAxis",
            oskar_sky_column_type_to_name(OSKAR_SKY_MAJOR_RAD, 0)
    );
    ASSERT_STREQ(
            "MinorAxis",
            oskar_sky_column_type_to_name(OSKAR_SKY_MINOR_RAD, 0)
    );
    ASSERT_STREQ(
            "Orientation",
            oskar_sky_column_type_to_name(OSKAR_SKY_PA_RAD, 0)
    );
    ASSERT_STREQ(
            "pa_deg",
            oskar_sky_column_type_to_name(OSKAR_SKY_PA_RAD, 1)
    );
    ASSERT_STREQ(
            "PolarizationAngle",
            oskar_sky_column_type_to_name(OSKAR_SKY_POLA_RAD, 0)
    );
    ASSERT_STREQ(
            "PolarizedFraction",
            oskar_sky_column_type_to_name(OSKAR_SKY_POLF, 0)
    );
    ASSERT_STREQ(
            "ReferenceWavelength",
            oskar_sky_column_type_to_name(OSKAR_SKY_REF_WAVE_M, 0)
    );
    ASSERT_STREQ(
            "SpectralCurvature",
            oskar_sky_column_type_to_name(OSKAR_SKY_SPEC_CURV, 0)
    );
    ASSERT_STREQ(
            "LineWidth",
            oskar_sky_column_type_to_name(OSKAR_SKY_LINE_WIDTH_HZ, 0)
    );
}
