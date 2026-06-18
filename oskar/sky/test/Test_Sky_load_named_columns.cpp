/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"
#include "utility/oskar_timer.h"

#define DEG2RAD (M_PI / 180.0)
#define ARCSEC2RAD (DEG2RAD / 3600.0)


static void check_test_sky_model(const oskar_Sky* sky)
{
    const double tol = 5e-14;
    double ra = 0., dec = 0., maj = 0., min = 0., pa = 0.;
    int s = 0;
    ra = 1.23456;
    dec = 43.21 * DEG2RAD;
    maj = 1. * ARCSEC2RAD;
    min = 2. * ARCSEC2RAD;
    pa = 3. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(99.9, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(+0.5, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 1;
    ra = 234.567 * DEG2RAD;
    dec = 54.32 * DEG2RAD;
    maj = 4. * ARCSEC2RAD;
    min = 5. * ARCSEC2RAD;
    pa = 6. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(799.9, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.7, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 2;
    ra = -15. * DEG2RAD * (1. / 60.0 + 1.1 / 3600.0);
    dec = -1. * DEG2RAD * (10.2 / 3600.0);
    maj = 5.1 * ARCSEC2RAD;
    min = 6.2 * ARCSEC2RAD;
    pa = 7.3 * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(34.56, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.7, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(100e3, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 3;
    ra = -15. * DEG2RAD * (2. / 60.0 + 3.4 / 3600.0);
    dec = -1. * DEG2RAD * (20.2 / 3600.0);
    maj = 5.2 * ARCSEC2RAD;
    min = 6.3 * ARCSEC2RAD;
    pa = 7.4 * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(45.67, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.7, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(101e3, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 4;
    maj = 6. * ARCSEC2RAD;
    min = 7. * ARCSEC2RAD;
    pa = 8.9 * DEG2RAD;
    ASSERT_NEAR(3.14, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(1.57, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(42., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(100e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 5;
    ASSERT_NEAR(3.14, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(1.57, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(42., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-0.2, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_MAJOR_RAD, s));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_MINOR_RAD, s));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_PA_RAD, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 6;
    ra = -15. * DEG2RAD * (1. + 2. / 60.0 + 3.456 / 3600.0);
    dec = -1. * DEG2RAD * (10. + 11. / 60.0 + 12.345 / 3600.0);
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(-0.123456789, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.0123456, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(1.23456, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(150e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 7;
    ra = 15. * DEG2RAD * (2. + 3. / 60.0 + 4.567 / 3600.0);
    dec = 1. * DEG2RAD * (11. + 12. / 60.0 + 13.456 / 3600.0);
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(1.234, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(0.7654, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0.3210, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(151e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 8;
    ra = 15. * DEG2RAD * (3. + 4. / 60.0 + 5.678 / 3600.0);
    dec = 1. * DEG2RAD * (12. + 13. / 60.0 + 14.567 / 3600.0);
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(2.345, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.6543, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-0.147, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(152e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 9;
    ra = 15. * DEG2RAD * (4. + 5. / 60.0 + 6.789 / 3600.0);
    dec = 1. * DEG2RAD * (13. + 14. / 60.0 + 15.678 / 3600.0);
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(3.456, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(0.0456, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0.0654, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(153e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 10;
    ra = 15. * DEG2RAD * (5. + 6. / 60.0 + 7.89 / 3600.0);
    dec = 1. * DEG2RAD * (14. + 15. / 60.0 + 16.789 / 3600.0);
    maj = 80.1 * ARCSEC2RAD;
    min = 50.2 * ARCSEC2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(4.567, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(-0.00321, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0.00123, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
    //----------------
    s = 11;
    ra = 15. * DEG2RAD * (6. + 7. / 60.0 + 8.9 / 3600.0);
    dec = 1. * DEG2RAD * (15. + 16. / 60.0 + 17.89 / 3600.0);
    maj = 741. * ARCSEC2RAD;
    min = 147. * ARCSEC2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(5.6789, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, s));
    ASSERT_EQ(9.876, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-1.234, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
}


TEST(Sky, load_named_columns_round_trip)
{
    const int type = OSKAR_DOUBLE;
    const char* file1 = "temp_test_sources1.txt";
    const char* file2 = "temp_test_sources_LOFAR.txt";
    const char* file3 = "temp_test_sources_SKA.txt";
    FILE* file = fopen(file1, "w");
    (void) fprintf(file, "# Some comments\n");
    (void) fprintf(file, "  #   (Name, Type ra dec I, SpectralIndex=-0.7, LogarithmicSI, ReferenceFrequency='123456.789', MajorAxis, MinorAxis, Orientation) = format\n");
    (void) fprintf(file, " # A pretty messy test file\n");
    (void) fprintf(file, "s0,   A,    1.23456rad '43.21deg'   99.9  +0.5 true,,1,2,3\n");
    (void) fprintf(file, "s1, B,   \"234.567deg\"   54.32deg 799.9,, true, , 4 5 6\n");
    (void) fprintf(file, "s2, C, -00:01:01.1 -00.00.10.2   34.56 [] false 100e3 5.1 6.2 7.3\n");
    (void) fprintf(file, " 'a quoted name, containing commas, and spaces'  C   -00h02m03.4 -00d00m20.2   45.67,[],1 101e3 5.2,6.3,7.4\n");
    (void) fprintf(file, ", D 3.14 1.57   42 0.1 true 100e6 6 7 8.9\n");
    (void) fprintf(file, ",,3.14 1.57   42 [0.1, -0.2, 0.3],,,,,\n");
    (void) fprintf(file, "alice,POINT, -01:02:03.456, -10.11.12.345, -0.123456789, [-0.0123456, 1.23456], false, 150e6,,,\n");
    (void) fprintf(file, "bob,POINT,02:03:04.567,11.12.13.456, 1.234,\"[0.7654,0.3210]\",false,151e6,,,\n");
    (void) fprintf(file, "charlie,POINT,03:04:05.678,12.13.14.567, 2.345,(-0.6543 -0.147),false,152e6,,,\n");
    (void) fprintf(file, "dave,POINT,04:05:06.789,13.14.15.678,3.456,'[0.0456  0.0654]',false,153e6,,,\n");
    (void) fprintf(file, "# A partially broken line (missing closing bracket; but quoted, so it should still work)\n");
    (void) fprintf(file, "eve,GAUSSIAN,\"05:06:07.89\",14.15.16.789, 4.567,\"[-0.00321,0.00123\",false,,80.1,50.2,0\n");
    (void) fprintf(file, "frank,GAUSSIAN,06:07:08.9,'15.16.17.89',5.6789,[9.876,-1.234],0,,741,147,0\n");
    (void) fclose(file);
    {
        // Load the test file and verify it.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file1, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        check_test_sky_model(sky);

        // Save it out to a new file with LOFAR conventions.
        oskar_sky_save_named_columns(
                sky, file2, false, false, true, true, false, true, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Save it out to a new file with SKA conventions.
        oskar_sky_save_named_columns(
                sky, file3, true, true, true, true, true, false, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_free(sky, &status);
    }
    {
        // Load back the LOFAR version and verify it again.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file2, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        check_test_sky_model(sky);
        oskar_sky_free(sky, &status);
    }
    {
        // Load back the SKA version and verify it again.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file3, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        check_test_sky_model(sky);
        oskar_sky_free(sky, &status);
    }
    (void) remove(file1);
    (void) remove(file2);
    (void) remove(file3);
}


TEST(Sky, load_named_columns_large_file)
{
    const int type = OSKAR_DOUBLE;
    const int num_sources = 101300;
    const char* file1 = "temp_test_sources1_large.txt";
    const char* file2 = "temp_test_sources2_large.txt";
    {
        FILE* file = fopen(file1, "w");
        (void) fprintf(file, "# A comment\n");
        (void) fprintf(file, "Format = Name, Type, Ra, Dec, I, Q, U, V, SpectralIndex='-0.7', LogarithmicSI, ReferenceFrequency='123456.789', MajorAxis, MinorAxis, Orientation\n");
        for (int i = 0; i < num_sources; ++i)
        {
            (void) fprintf(
                    file, "s%d, GAUSSIAN, %.16gdeg, %.16gdeg, "
                    "%.16g, %.16g, %.16g, %.16g,,"
                    "true,,%16g, %.16g, %.16g\n", i,
                    360.0 * i / (double) num_sources,
                    90.0 * i / (double) num_sources,
                    3.3 * i, 4.4 * i, 5.5 * i, 6.6 * i, 7.7, 8.8, 9.9
            );
        }
        (void) fclose(file);
    }
    {
        // Load the test file and verify it.
        int status = 0;
        oskar_Timer* timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
        oskar_timer_start(timer);
        oskar_Sky* sky = oskar_sky_load_named_columns(file1, type, &status);
        printf("Loaded %d sources in %.3g sec\n",
                num_sources, oskar_timer_elapsed(timer)
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        for (int i = 0; i < num_sources; ++i)
        {
            ASSERT_NEAR(360.0 * i / (double) num_sources * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i), 5e-15
            );
            ASSERT_NEAR(90.0 * i / (double) num_sources * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i), 5e-15
            );
            ASSERT_DOUBLE_EQ(3.3 * i,
                    oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(4.4 * i,
                    oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(5.5 * i,
                    oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(6.6 * i,
                    oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(-0.7,
                    oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, i)
            );
            ASSERT_DOUBLE_EQ(123456.789,
                    oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, i)
            );
            ASSERT_DOUBLE_EQ(7.7 * ARCSEC2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(8.8 * ARCSEC2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(9.9 * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, i)
            );
        }

        // Save it out to a new file.
        oskar_timer_start(timer);
        oskar_sky_save_named_columns(
                sky, file2, false, true, true, true, false, true, &status
        );
        printf("Saved %d sources in %.3g sec\n",
                num_sources, oskar_timer_elapsed(timer)
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_free(sky, &status);
        oskar_timer_free(timer);
    }
    {
        // Load it back and verify it again.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file2, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        for (int i = 0; i < num_sources; ++i)
        {
            ASSERT_NEAR(360.0 * i / (double) num_sources * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i), 5e-15
            );
            ASSERT_NEAR(90.0 * i / (double) num_sources * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i), 5e-15
            );
            ASSERT_DOUBLE_EQ(3.3 * i,
                    oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(4.4 * i,
                    oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(5.5 * i,
                    oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(6.6 * i,
                    oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, i)
            );
            ASSERT_DOUBLE_EQ(-0.7,
                    oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, i)
            );
            ASSERT_DOUBLE_EQ(123456.789,
                    oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, i)
            );
            ASSERT_DOUBLE_EQ(7.7 * ARCSEC2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(8.8 * ARCSEC2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(9.9 * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, i)
            );
        }
        oskar_sky_free(sky, &status);
    }
    (void) remove(file1);
    (void) remove(file2);
}


TEST(Sky, load_named_columns_degrees)
{
    int status = 0;
    const char* name = "temp_test_degree_columns.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, " Format=RaD, DecD, I\n");
    (void) fprintf(file, "1, 2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(2. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ska_convention)
{
    int status = 0;
    const char* name = "temp_test_ska_convention.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra_deg, dec_deg, i_pol_jy, ref_freq_hz, spec_idx) = format\n");
    (void) fprintf(file, "1, 2, 3, 100e6, \"[0.1,0.2,,,]\"\n");
    (void) fprintf(file, "4, 5, 6, 200e6, \"[0.3,,,,]\"\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(2. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(100e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 0));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 0));
    ASSERT_EQ(4. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(5. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(6., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(200e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 1));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 1));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_different_order)
{
    int status = 0;
    const char* name = "temp_test_different_order.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# Format = I,ReferenceFrequency,Dec,RA\n");
    (void) fprintf(file, "3, 200e6, -1.570796, 1\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(-1.570796, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(200e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_mixed_units)
{
    int status = 0;
    const char* name = "temp_test_mixed_units.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "#  Format  =  RA, Dec,I\n");
    (void) fprintf(file, "1.1deg, -1.2rad, 3.14\n");
    (void) fprintf(file, "2.2rad, -3.4deg, 2.718\n");
    (void) fprintf(file, "0.123, 0.234deg, 10.01\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(-1.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.14, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(2.2, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(-3.4 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(2.718, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(0.123, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 2));
    ASSERT_EQ(0.234 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 2));
    ASSERT_EQ(10.01, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 2));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_format_at_end_without_brackets)
{
    int status = 0;
    const char* name = "temp_test_format_at_end_without_brackets.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "RA, Dec, I = format\n");
    (void) fprintf(file, " 0.1, 0.2, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_all_defaults_different_quote_styles)
{
    int status = 0;
    const char* name = "temp_test_all_defaults_different_quote_styles.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "Format=RA=`0.1`, Dec='0.2', I=\"3.3\"\n");
    (void) fprintf(file, ",,\n");
    (void) fprintf(file, "0.5, 0.6,\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(0.5, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.6, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_multiple_defaults_with_comments)
{
    int status = 0;
    const char* name = "temp_test_multiple_defaults_with_comments.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RA, Dec, I=10.5, Q, U='1.2', V) = format\n");
    (void) fprintf(file, "0.1,0.2,3,,,\n");
    (void) fprintf(file, "0.3 0.4,,,,\n");
    (void) fprintf(file, "# embedded comment\n");
    (void) fprintf(file, "0.5, 0.6, 300, 400, 500, 600\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "0.7, 0.8,,,,\n");
    (void) fprintf(file, "0.9, 1.0, , , ,\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(5, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 0));
    ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 0));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 0));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.4, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 1));
    ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 1));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 1));
    ASSERT_EQ(0.5, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 2));
    ASSERT_EQ(0.6, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 2));
    ASSERT_EQ(300., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 2));
    ASSERT_EQ(400., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 2));
    ASSERT_EQ(500., oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 2));
    ASSERT_EQ(600., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 2));
    ASSERT_EQ(0.7, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 3));
    ASSERT_EQ(0.8, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 3));
    ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 3));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 3));
    ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 3));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 3));
    ASSERT_EQ(0.9, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 4));
    ASSERT_EQ(1.0, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 4));
    ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 4));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 4));
    ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 4));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 4));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_logarithmic_si_implicit_true)
{
    /* LogarithmicSI should be implicitly true by default. */
    int status = 0;
    const char* name = "temp_test_logarithmic_si_implicit_true_by_default.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# ( RA, Dec, I, RefFreq=100e6, SpectralIndex, LogarithmicSI ) = format\n");
    (void) fprintf(file, "0.123, 0.234, 3,, -0.5, false\n");
    (void) fprintf(file, "0.345, 0.456, 30,, -0.6,\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.123, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.234, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(-0.5, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 0));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 0));
    ASSERT_EQ(0.345, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.456, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(-0.6, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 1));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_logarithmic_si_default_false)
{
    int status = 0;
    const char* name = "temp_test_logarithmic_si_default_false.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RA, Dec, I, RefFreq=100e6, SpectralIndex, LogarithmicSI='false') = format\n");
    (void) fprintf(file, "0.123, 0.234, 3,, -0.5, true\n");
    (void) fprintf(file, "0.345, 0.456, 30,, -0.6,\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.123, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.234, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(-0.5, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 0));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 0));
    ASSERT_EQ(0.345, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.456, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(-0.6, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 1));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ignore_dummy_column_spaces)
{
    int status = 0;
    const char* name = "temp_test_ignore_dummy_column_spaces.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RA Dec dummy    I) = format\n");
    (void) fprintf(file, "0.1 0.2 123456.789 3.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ignore_dummy_column_commas)
{
    int status = 0;
    const char* name = "temp_test_ignore_dummy_column_commas.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RA,Dec='0.2', dummy,I) = format\n");
    (void) fprintf(file, "0.1, , 123456.789, 3.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ignore_empty_column)
{
    int status = 0;
    const char* name = "temp_test_ignore_empty_column.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RA,Dec,,I) = format\n");
    (void) fprintf(file, "0.1, 0.2, 123456.789, 3.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_format_in_comments)
{
    int status = 0;
    const char* name = "temp_test_format_in_comments.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# [rad, decd, i] = format     \n");
    (void) fprintf(file, "# Should be able to use the word 'format'\n");
    (void) fprintf(file, "# in comments after the format line.\n");
    (void) fprintf(file, "1.1, 2.2, 3.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(2.2 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_different_ra_dec_labels)
{
    int status = 0;
    const char* name = "temp_test_different_ra_dec_labels.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "format = RAJ2000 DEJ2000, I\n");
    (void) fprintf(file, "1.1deg 2.2deg, 3.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(2.2 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_vector_default_double_quotes)
{
    int status = 0;
    const char* name = "temp_test_vector_default_double_quotes.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "Format=Ra Dec I=\"[0.1 0.2]\" RefFreq = \"[100e6 101e6]\"\n");
    (void) fprintf(file, "0.1, 0.2,,\n");
    (void) fprintf(file, "0.3, 0.4, 5, 102e6\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 0));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 0));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
    ASSERT_EQ(100e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 0));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.4, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 1));
    ASSERT_EQ(5., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
    ASSERT_EQ(102e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
    ASSERT_EQ(0, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_vector_default_single_quotes)
{
    int status = 0;
    const char* name = "temp_test_vector_default_single_quotes.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I='[0.1,0.2]' RefFreq = `[100e6,101e6]`) = format\n");
    (void) fprintf(file, "0.1, 0.2,,\n");
    (void) fprintf(file, "0.3, 0.4, 5, 102e6\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 0));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 0));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
    ASSERT_EQ(100e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 0));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.4, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 1));
    ASSERT_EQ(5., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
    ASSERT_EQ(102e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
    ASSERT_EQ(0, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_implicit_reference_freq)
{
    int status = 0;
    const char* name = "temp_test_implicit_reference_freq.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra,        Dec  , Fint12  , Fint23  , Fint34) = format\n");
    (void) fprintf(file, "  0.1      ,  0.2  , \"12.1\", \"23.2\", 34.3\n");
    (void) fprintf(file, "\"0.3E+00\",\"0.4\",   45.4      56.5    67.6\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(8, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 0));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 1));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(12.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(23.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
    ASSERT_EQ(34.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 0));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.4, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(45.4, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(56.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
    ASSERT_EQ(67.6, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 1));
    // Reference frequency columns should have also been created.
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 0));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 1));
    ASSERT_EQ(12e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    ASSERT_EQ(23e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 0));
    ASSERT_EQ(34e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 2, 0));
    ASSERT_EQ(12e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
    ASSERT_EQ(23e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 1));
    ASSERT_EQ(34e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 2, 1));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_mixed_spectral_types)
{
    int status = 0;
    const char* name = "temp_test_mixed_spectral_types.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "(Ra,  Dec, I,                     ReferenceFrequency                , FrequencyIncrement, SpectralIndex     , LogarithmicSI, SpectralCurvature, LineWidth) = format\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 0: Flat spectrum (no reference frequency).\n");
    (void) fprintf(file, "0.00, 0.0, 1.0,                                                     ,                   ,                   ,              ,                  ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 1: Simple logarithmic spectral index.\n");
    (void) fprintf(file, "0.01, 0.1, 1.1,                   101e6                             ,                   , -0.55             ,              ,                  ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 2: Two-term logarithmic spectral index polynomial.\n");
    (void) fprintf(file, "0.02, 0.2, 1.2,                   102e6                             ,                   , [-0.7, 0.05]      , true         ,                  ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 3: Three-term linear spectral index polynomial.\n");
    (void) fprintf(file, "0.03, 0.3, 1.3,                   103e6                             ,                   , [0.08, 0.07, 0.02], false        ,                  ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 4: Spectral curvature model.\n");
    (void) fprintf(file, "0.04, 0.4, 1.4,                   [104e6]                           ,                   , [-0.6]            ,              , 0.1              ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 5: Simple Gaussian spectral line model.\n");
    (void) fprintf(file, "0.05, 0.5, 1.5,                   105e6                             ,                   ,                   ,              ,                  , 100e3\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 6: Three spectral lines of the same width, each a Gaussian.\n");
    (void) fprintf(file, "0.06, 0.6, [1.6, 1.7, 1.8],       [101e6, 102e6, 104e6]             ,                   ,                   ,              ,                  , 125e3\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 7: Three spectral lines of different widths, each a Gaussian.\n");
    (void) fprintf(file, "0.07, 0.7, [1.6, 1.7, 1.8],       [101e6, 102e6, 104e6]             ,                   ,                   ,              ,                  , [250e3, 350e3, 500e3]\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 8: Different flux at four arbitrary frequencies.\n");
    (void) fprintf(file, "0.08, 0.8, [1.7, 1.8, 1.9, 1.75], [101e6, 102.4e6, 103.8e6, 104.1e6],                   ,                   ,              ,                  ,\n");
    (void) fprintf(file, "\n");
    (void) fprintf(file, "# Source 9: Different flux at four regularly-spaced frequencies.\n");
    (void) fprintf(file, "0.09, 0.9, [1.5, 1.6, 1.7, 1.55], 101e6                             , 1e6               ,                   ,              ,                  ,\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(19, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(10, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(0.00, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
    ASSERT_EQ(0.01, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
    ASSERT_EQ(0.02, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 2));
    ASSERT_EQ(0.03, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 3));
    ASSERT_EQ(0.04, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 4));
    ASSERT_EQ(0.05, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 5));
    ASSERT_EQ(0.06, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 6));
    ASSERT_EQ(0.07, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 7));
    ASSERT_EQ(0.08, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 8));
    ASSERT_EQ(0.09, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
    ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 2));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 3));
    ASSERT_EQ(0.4, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 4));
    ASSERT_EQ(0.5, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 5));
    ASSERT_EQ(0.6, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 6));
    ASSERT_EQ(0.7, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 7));
    ASSERT_EQ(0.8, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 8));
    ASSERT_EQ(0.9, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 9));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 0));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 2));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 3));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 4));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 5));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 6));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 7));
    ASSERT_EQ(4, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 8));
    ASSERT_EQ(4, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_I_JY, 9));
    ASSERT_EQ(1.0, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
    ASSERT_EQ(1.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
    ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 2));
    ASSERT_EQ(1.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 3));
    ASSERT_EQ(1.4, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 4));
    ASSERT_EQ(1.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 5));
    ASSERT_EQ(1.6, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 6));
    ASSERT_EQ(1.7, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 6));
    ASSERT_EQ(1.8, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 6));
    ASSERT_EQ(1.6, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 7));
    ASSERT_EQ(1.7, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 7));
    ASSERT_EQ(1.8, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 7));
    ASSERT_EQ(1.7, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 8));
    ASSERT_EQ(1.8, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 8));
    ASSERT_EQ(1.9, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 8));
    ASSERT_EQ(1.75, oskar_sky_data(sky, OSKAR_SKY_I_JY, 3, 8));
    ASSERT_EQ(1.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 9));
    ASSERT_EQ(1.6, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 9));
    ASSERT_EQ(1.7, oskar_sky_data(sky, OSKAR_SKY_I_JY, 2, 9));
    ASSERT_EQ(1.55, oskar_sky_data(sky, OSKAR_SKY_I_JY, 3, 9));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 0));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 1));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 2));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 3));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 4));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 5));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 6));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 7));
    ASSERT_EQ(4, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 8));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_REF_HZ, 9));
    ASSERT_EQ(0, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
    ASSERT_EQ(102e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 2));
    ASSERT_EQ(103e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 3));
    ASSERT_EQ(104e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 4));
    ASSERT_EQ(105e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 5));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 6));
    ASSERT_EQ(102e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 6));
    ASSERT_EQ(104e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 2, 6));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 7));
    ASSERT_EQ(102e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 7));
    ASSERT_EQ(104e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 2, 7));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 8));
    ASSERT_EQ(102.4e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 1, 8));
    ASSERT_EQ(103.8e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 2, 8));
    ASSERT_EQ(104.1e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 3, 8));
    ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 9));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 0));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 1));
    ASSERT_EQ(2, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 2));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 3));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 4));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 5));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 6));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 7));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 8));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_IDX, 9));
    ASSERT_EQ(0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 0));
    ASSERT_EQ(-0.55, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 1));
    ASSERT_EQ(-0.7, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 2));
    ASSERT_EQ(0.05, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 2));
    ASSERT_EQ(0.08, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 3));
    ASSERT_EQ(0.07, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 3));
    ASSERT_EQ(0.02, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, 3));
    ASSERT_EQ(-0.6, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 4));
    ASSERT_EQ(0.0,  oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 4));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 5));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 5));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 6));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 6));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 7));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 7));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, 9));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 0));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 1));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 2));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 3));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 4));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 5));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 6));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 7));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 8));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_SPEC_CURV, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 0));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 1));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 2));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 3));
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 4));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 5));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 6));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 7));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_SPEC_CURV, 0, 9));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 1));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 2));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 3));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 4));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 5));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 6));
    ASSERT_EQ(3, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 7));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 8));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_LINE_WIDTH_HZ, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 0));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 1));
    ASSERT_EQ(100e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 5));
    ASSERT_EQ(125e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 6));
    ASSERT_EQ(250e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 7));
    ASSERT_EQ(350e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 1, 7));
    ASSERT_EQ(500e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 2, 7));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 1, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 2, 8));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 1, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 2, 9));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 0));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 1));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 2));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 3));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 4));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 5));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 6));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 7));
    ASSERT_EQ(0, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 8));
    ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(sky, OSKAR_SKY_INC_HZ, 9));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 0));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 1));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 2));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 3));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 4));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 5));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 6));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 7));
    ASSERT_EQ(0.0, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 8));
    ASSERT_EQ(1e6, oskar_sky_data(sky, OSKAR_SKY_INC_HZ, 0, 9));
    oskar_sky_free(sky, &status);
    (void) remove(name);
}


TEST(Sky, load_named_columns_broken_format_at_end)
{
    int status = 0;
    const char* name = "temp_test_broken_format_at_end.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra,dec,i) format\n");
    (void) fprintf(file, "0.1, 0.2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_broken_format_at_start)
{
    int status = 0;
    const char* name = "temp_test_broken_format_at_start.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "Format RA, Dec, I\n");
    (void) fprintf(file, "0.1, 0.2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_broken_format_extra_chars)
{
    int status = 0;
    const char* name = "temp_test_broken_format_extra_chars.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra,dec,i) = format blah\n");
    (void) fprintf(file, "0.1, 0.2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_broken_first_line)
{
    int status = 0;
    const char* name = "temp_test_broken_first_line.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra,dec,i,referencefrequency) = format\n");
    (void) fprintf(file, "0.1, 0.2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_broken_middle_line)
{
    int status = 0;
    const char* name = "temp_test_broken_middle_line.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra,dec,i,q,u,v) = format\n");
    (void) fprintf(file, "# some comments to check that the error\n");
    (void) fprintf(file, "# is printed for the correct line (line 5)\n");
    (void) fprintf(file, "0.1, 0.2, 3 4 5 6\n");
    (void) fprintf(file, "0.1, 0.2, 3 4 5 6,false\n");
    (void) fprintf(file, "0.1, 0.2, 3 4 5 6\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_closing_bracket)
{
    int status = 0;
    const char* name = "temp_test_missing_closing_bracket.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (ra,dec,i,spectralindex) = format\n");
    (void) fprintf(file, "0.1, 0.2, 3 [0.1, 0.2\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_flux)
{
    int status = 0;
    const char* name = "temp_test_missing_flux.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec) = format\n");
    (void) fprintf(file, "0.1, 0.2\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_dec)
{
    int status = 0;
    const char* name = "temp_test_missing_dec.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra,I) = format\n");
    (void) fprintf(file, "0.1,0.2\n");
    (void) fprintf(file, "0.3,0.4\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_ra_dec)
{
    int status = 0;
    const char* name = "temp_test_missing_ra_dec.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (I) = format\n");
    (void) fprintf(file, "1\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_multiple_ra_columns)
{
    int status = 0;
    const char* name = "temp_test_multiple_ra_columns.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, RaD, I) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_reference_freq_for_multiple_flux)
{
    int status = 0;
    const char* name = "temp_test_missing_reference_freq_for_multiple_flux.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3, 0.4], [100e6]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_multiple_flux_for_reference_freq)
{
    int status = 0;
    const char* name = "temp_test_missing_multiple_flux_for_reference_freq.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, [10e6, 11e6]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_reference_freq_for_spectral_index)
{
    int status = 0;
    const char* name = "temp_test_missing_reference_freq_for_spectral_index.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3,, -0.7\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_too_many_fluxes_for_spectral_index)
{
    int status = 0;
    const char* name = "temp_test_too_many_fluxes_for_spectral_index.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3, 0.4], 100e6, -0.7\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_too_many_rm)
{
    int status = 0;
    const char* name = "temp_test_too_many_rm.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, RotationMeasure) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, [0.25, 0.5]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_spectral_index_with_log_si)
{
    int status = 0;
    const char* name = "temp_test_missing_spectral_index_with_log_si.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex, LogarithmicSI) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, 0.2, false\n");
    (void) fprintf(file, "0.11, 0.22, 0.33, 100e6,, false\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_too_many_ref_wavelengths)
{
    int status = 0;
    const char* name = "temp_test_too_many_ref_wavelengths.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, RotationMeasure, ReferenceWavelength) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, 0.2, 0.1\n");
    (void) fprintf(file, "0.11, 0.22, [0.33 0.44], [100e6 101e6], 0.15, [1.1, 1.2]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_too_many_spectral_curvature)
{
    int status = 0;
    const char* name = "temp_test_too_many_spectral_curvature.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex, SpectralCurvature) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, 0.2, 0.1\n");
    (void) fprintf(file, "0.11, 0.22, 0.33, 100e6, 0.15, [1.1, 1.2]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_spectral_curvature_with_no_spectral_index)
{
    int status = 0;
    const char* name = "temp_test_spectral_curvature_with_no_spectral_index.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralCurvature) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, 0.1\n");
    (void) fprintf(file, "0.11, 0.22, 0.33, 100e6, 0.15\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_spectral_curvature_with_too_many_spectral_index)
{
    int status = 0;
    const char* name = "temp_test_spectral_curvature_with_too_many_spectral_index.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex, SpectralCurvature) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, [-0.7], 0.1\n");
    (void) fprintf(file, "0.11, 0.22, 0.33, 100e6, [-0.5, 0.01], 0.15\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_inconsistent_pol_angle)
{
    int status = 0;
    const char* name = "temp_test_inconsistent_pol_angle.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, PolarisationAngle) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, [100e6], [23.0]\n");
    (void) fprintf(file, "0.11, 0.22, [0.33, 0.44], [100e6, 101e6], [-0.5, 0.01]\n");
    (void) fprintf(file, "0.111, 0.222, [0.333, 0.444], [100e6, 101e6], [-11 -22 -33]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_too_many_freq_inc)
{
    int status = 0;
    const char* name = "temp_test_too_many_freq_inc.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, FreqInc) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, [100e6], [1e6]\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, [100e6], [100e3, 500e3]\n");
    (void) fprintf(file, "0.11, 0.22, [0.33, 0.44], [100e6, 101e6], [1e3, 1e4]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_freq_inc_with_multiple_ref_freq)
{
    int status = 0;
    const char* name = "temp_test_freq_inc_with_multiple_ref_freq.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, FreqInc) = format\n");
    (void) fprintf(file, "0.11, 0.22, [0.33, 0.44], [100e6, 101e6], 1e3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_inconsistent_pol_frac)
{
    int status = 0;
    const char* name = "temp_test_inconsistent_pol_frac.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, PolarizedFraction) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, [100e6], [0.05]\n");
    (void) fprintf(file, "0.11, 0.22, [0.33, 0.44], [100e6, 101e6], [0.01, 0.02]\n");
    (void) fprintf(file, "0.111, 0.222, [0.333, 0.444], [100e6, 101e6], [0.04, 0.05, 0.06]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_inconsistent_line_width)
{
    int status = 0;
    const char* name = "temp_test_inconsistent_line_width.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, LineWidth) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3], [100e6], [10e3]\n");
    (void) fprintf(file, "0.11, 0.22, [0.33, 0.44], [100e6, 101e6], [11e3, 12e3]\n");
    (void) fprintf(file, "0.111, 0.222, [0.333, 0.444, 0.555], [100e6, 101e6, 102e6], [13e3, 14e3]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_inconsistent_line_width_and_flux)
{
    int status = 0;
    const char* name = "temp_test_inconsistent_line_width_and_flux.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, LineWidth) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3], [100e6], [10e3]\n");
    (void) fprintf(file, "0.11, 0.22, [0.33], [100e6, 101e6], [11e3, 12e3]\n");
    (void) fprintf(file, "0.111, 0.222, [0.333, 0.444], [100e6, 101e6], [13e3]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_line_width_and_spectral_index)
{
    int status = 0;
    const char* name = "temp_test_line_width_and_spectral_index.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, LineWidth, SpectralIndex) = format\n");
    (void) fprintf(file, "0.1, 0.2, 0.3, 100e6, 10e3, -0.7\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_unsupported_dimensions)
{
    int status = 0;
    const char* name = "temp_test_unsupported_dimensions.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, Major, Minor, Orientation) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3, 0.4], [100e6, 101e6], [0.25, 0.5], [0.15, 0.45], [10, 20]\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_spectral_index_and_multiple_flux)
{
    int status = 0;
    const char* name = "temp_test_spectral_index_and_multiple_flux.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, SpectralIndex) = format\n");
    (void) fprintf(file, "0.1, 0.2, [0.3, 0.4], [10e6, 11e6], -0.7\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ra_radians_out_of_range)
{
    int status = 0;
    const char* name = "temp_test_ra_radians_out_of_range.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I) = format\n");
    (void) fprintf(file, "270.0, 0.2, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_ra_degrees_out_of_range)
{
    int status = 0;
    const char* name = "temp_test_ra_degrees_out_of_range.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (RaD, Dec, I) = format\n");
    (void) fprintf(file, "361.0, 0.2, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_dec_radians_out_of_range)
{
    int status = 0;
    const char* name = "temp_test_dec_radians_out_of_range.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, Dec, I) = format\n");
    (void) fprintf(file, "0.1, 45.0, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_dec_degrees_out_of_range)
{
    int status = 0;
    const char* name = "temp_test_dec_degrees_out_of_range.txt";
    FILE* file = fopen(name, "w");
    (void) fprintf(file, "# (Ra, DecD, I) = format\n");
    (void) fprintf(file, "0.1, 91.0, 0.3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}


TEST(Sky, load_named_columns_missing_format_line)
{
    int status = 0;
    const char* name = "temp_test_missing_format_line.txt";
    FILE* file = fopen(name, "w");
    // Don't include the word "format" in the header, otherwise it will
    // assume it's malformed and return an invalid argument error.
    (void) fprintf(file, "# no special header\n");
    (void) fprintf(file, "1, 2, 3\n");
    (void) fclose(file);
    oskar_Sky* sky = oskar_sky_load_named_columns(name, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
    ASSERT_EQ(0, sky);
    (void) remove(name);
}
