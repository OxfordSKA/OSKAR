/*
 * Copyright (c) 2025, The OSKAR Developers.
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
    ra = 123.456;
    dec = 43.21 * DEG2RAD;
    maj = 1. * ARCSEC2RAD;
    min = 2. * ARCSEC2RAD;
    pa = 3. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(99.9, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(0.7, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 1;
    ra = 123.456 * DEG2RAD;
    dec = 43.21 * DEG2RAD;
    maj = 4. * ARCSEC2RAD;
    min = 5. * ARCSEC2RAD;
    pa = 6. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(99.9, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
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
    maj = 5. * ARCSEC2RAD;
    min = 6. * ARCSEC2RAD;
    pa = 7. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(34.56, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(+0.5, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(100e3, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_NEAR(pa, oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s), tol);
    //----------------
    s = 3;
    ra = -15. * DEG2RAD * (1. / 60.0 + 1.1 / 3600.0);
    dec = -1. * DEG2RAD * (10.2 / 3600.0);
    maj = 5. * ARCSEC2RAD;
    min = 6. * ARCSEC2RAD;
    pa = 7. * DEG2RAD;
    ASSERT_NEAR(ra, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, s), tol);
    ASSERT_NEAR(dec, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, s), tol);
    ASSERT_EQ(34.56, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, s));
    ASSERT_EQ(+0.5, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(100e3, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
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
    ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-0.2, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0.3, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
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
    ASSERT_EQ(-0.7654, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-0.3210, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
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
    ASSERT_EQ(9.876, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, s));
    ASSERT_EQ(-1.234, oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 1, s));
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 2, s));
    ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, s));
    ASSERT_EQ(123456.789, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, s));
    ASSERT_NEAR(maj, oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, s), tol);
    ASSERT_NEAR(min, oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, s), tol);
    ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, s));
}


TEST(Sky, named_columns_load_save)
{
    const int type = OSKAR_DOUBLE;

    // Create a test file to load.
    const char* file1 = "temp_test_sources1.txt";
    const char* file2 = "temp_test_sources2.txt";
    FILE* file = fopen(file1, "w");
    (void) fprintf(file, "# Some comments\n");
    (void) fprintf(file, "  #   (Name, Type ra dec I, SpectralIndex=-0.7, LogarithmicSI, ReferenceFrequency='123456.789', MajorAxis, MinorAxis, Orientation) = format\n");
    (void) fprintf(file, " # A pretty messy test file\n");
    (void) fprintf(file, "s0,   A,    123.456rad 43.21deg   99.9  +0.7 true,,1,2,3\n");
    (void) fprintf(file, "s1, B, 123.456deg 43.21deg 99.9,, true, ,4 5 6\n");
    (void) fprintf(file, "s2, C, -00:01:01.1 -00.00.10.2   34.56 +0.5 true 100e3 5,6,7\n");
    (void) fprintf(file, "s3, C, -00h01m01.1 -00d00m10.2   34.56 +0.5 true 100e3 5,6,7\n");
    (void) fprintf(file, ", D 3.14 1.57   42 0.1 true 100e6 6 7 8.9\n");
    (void) fprintf(file, ",,3.14 1.57   42 [0.1, -0.2, 0.3]\n");
    (void) fprintf(file, "alice,POINT, -01:02:03.456, -10.11.12.345, -0.123456789, [-0.0123456, 1.23456], false, 150e6,,,\n");
    (void) fprintf(file, "bob,POINT,02:03:04.567,11.12.13.456, 1.234,[0.7654,0.3210],false,151e6,,,\n");
    (void) fprintf(file, "charlie,POINT,03:04:05.678,12.13.14.567, 2.345,[-0.7654 -0.3210],false,152e6,,,\n");
    (void) fprintf(file, "dave,POINT,04:05:06.789,13.14.15.678,3.456,[0.0456  0.0654],false,153e6,,,\n");
    (void) fprintf(file, "eve,GAUSSIAN,05:06:07.89,14.15.16.789, 4.567,[-0.00321,0.00123],false,,80.1,50.2,0\n");
    (void) fprintf(file, "frank,GAUSSIAN,06:07:08.9,15.16.17.89,5.6789,[9.876,-1.234],false,,741,147,0\n");
    (void) fclose(file);

    {
        // Load the test file and verify it.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file1, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        check_test_sky_model(sky);

        // Save it out to a new file.
        oskar_sky_save_named_columns(sky, file2, true, true, true, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_free(sky, &status);
    }
    {
        // Load it back and verify it again.
        int status = 0;
        oskar_Sky* sky = oskar_sky_load_named_columns(file2, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        check_test_sky_model(sky);
        oskar_sky_free(sky, &status);
    }

    // Clean up.
    (void) remove(file1);
    (void) remove(file2);
}


TEST(Sky, named_columns_load_save_large_file)
{
    const int type = OSKAR_DOUBLE;

    // Create a test file to load.
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
                    1.1 * i, 2.2 * i, 3.3 * i, 4.4 * i, 5.5 * i, 6.6 * i,
                    7.7, 8.8, 9.9
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
            ASSERT_DOUBLE_EQ(1.1 * i * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(2.2 * i * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i)
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
        oskar_sky_save_named_columns(sky, file2, true, true, true, &status);
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
            ASSERT_DOUBLE_EQ(1.1 * i * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i)
            );
            ASSERT_DOUBLE_EQ(2.2 * i * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i)
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

    // Clean up.
    (void) remove(file1);
    (void) remove(file2);
}


TEST(Sky, named_columns_different_format_lines)
{
    const char* filename = "temp_test_sources.txt";
    const int type = OSKAR_DOUBLE;
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "Format = RA, Dec, I\n");
        (void) fprintf(file, "1deg, 2deg, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, " Format=RaD, DecD, I\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# Format = I,RA,Dec\n");
        (void) fprintf(file, "3, 1, 2\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "#  Format  =  RA, Dec,I\n");
        (void) fprintf(file, "1deg, -1rad, 3.14\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1. * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(-1., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.14, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA, Dec, I) = format\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "RA, Dec, I = format\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA, Dec, I='12') = format\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fprintf(file, "10, 20\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(12., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "Format = RA, Dec, I='12'\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fprintf(file, "10, 20\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(12., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "Format = RA, Dec, I=12\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fprintf(file, "10, 20\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(12., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "Format=RA=1.1, Dec=2.2, I=3.3\n");
        (void) fprintf(file, ",,\n");
        (void) fprintf(file, "10, 20\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1.1, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA, Dec, I=10.5, Q, U='1.2', V) = format\n");
        (void) fprintf(file, "1,2,3\n");
        (void) fprintf(file, "10 20\n");
        (void) fprintf(file, "# embedded comment\n");
        (void) fprintf(file, "100, 200, 300, 400, 500, 600\n");
        (void) fprintf(file, "\n");
        (void) fprintf(file, "1000, 2000,,,,\n");
        (void) fprintf(file, "10000, 20000, , , ,\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(5, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 0));
        ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 0));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 1));
        ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 1));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 1));
        ASSERT_EQ(100., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 2));
        ASSERT_EQ(200., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 2));
        ASSERT_EQ(300., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 2));
        ASSERT_EQ(400., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 2));
        ASSERT_EQ(500., oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 2));
        ASSERT_EQ(600., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 2));
        ASSERT_EQ(1000., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 3));
        ASSERT_EQ(2000., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 3));
        ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 3));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 3));
        ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 3));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 3));
        ASSERT_EQ(10000., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 4));
        ASSERT_EQ(20000., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 4));
        ASSERT_EQ(10.5, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 4));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, 4));
        ASSERT_EQ(1.2, oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, 4));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, 4));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        /* LogarithmicSI should be implicitly true by default. */
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# ( RA, Dec, I, LogarithmicSI ) = format\n");
        (void) fprintf(file, "1, 2, 3, false\n");
        (void) fprintf(file, "10, 20, 30\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA, Dec, I, LogarithmicSI='false') = format\n");
        (void) fprintf(file, "1, 2, 3, true\n");
        (void) fprintf(file, "10, 20, 30\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_LIN_SI, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA Dec='2' dummy I) = format\n");
        (void) fprintf(file, "1, , 123456.789, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA,Dec,,I) = format\n");
        (void) fprintf(file, "1, 2, 123456.789, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA, Dec, , I) = format\n");
        (void) fprintf(file, "1, 2, 123456.789, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA,Dec, dummy,I) = format\n");
        (void) fprintf(file, "1, 2, 123456.789, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (RA Dec    I) = format\n");
        (void) fprintf(file, "1.1deg 2.2 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2.2, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "(ra,dec,i)=format\n");
        (void) fprintf(file, "1, 2, [3.01]\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.01, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "(ra,dec,i)=format\n");
        (void) fprintf(file, "1, 2, [3.01, 30.1]\n");
        (void) fprintf(file, "10 20 [4.01, 40.1]");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.01, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(30.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
        ASSERT_EQ(10., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(20., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(4.01, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        ASSERT_EQ(40.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# [rad, decd, i] = format     \n");
        (void) fprintf(file, "# Should be able to use the word 'format'\n");
        (void) fprintf(file, "# in comments after the format line.\n");
        (void) fprintf(file, "1.1, 2.2, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2.2 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "format = RAJ2000 DEJ2000, I\n");
        (void) fprintf(file, "1.1deg 2.2deg, 3.3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(3, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(1, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1.1 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2.2 * DEG2RAD, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3.3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, Dec, I=\"[0.1,0.2]\") = format\n");
        (void) fprintf(file, "1, 2\n");
        (void) fprintf(file, "3, 4, 5\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(4., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(5., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, Dec, I='[0.1, 0.2]') = format\n");
        (void) fprintf(file, "1, 2\n");
        (void) fprintf(file, "3, 4, 5\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(4, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(0.1, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(0.2, oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(4., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(5., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        ASSERT_EQ(0., oskar_sky_data(sky, OSKAR_SKY_I_JY, 1, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, Dec, I, ReferenceFrequency, LineWidth) = format\n");
        (void) fprintf(file, "1, 2, 3, 100e6, 100e3\n");
        (void) fprintf(file, "3, 4, 5, 101e6, 150e3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(5, oskar_sky_int(sky, OSKAR_SKY_NUM_COLUMNS));
        ASSERT_EQ(2, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        ASSERT_EQ(1., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 0));
        ASSERT_EQ(2., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 0));
        ASSERT_EQ(3, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 0));
        ASSERT_EQ(100e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 0));
        ASSERT_EQ(100e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 0));
        ASSERT_EQ(3., oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, 1));
        ASSERT_EQ(4., oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, 1));
        ASSERT_EQ(5., oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, 1));
        ASSERT_EQ(101e6, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, 1));
        ASSERT_EQ(150e3, oskar_sky_data(sky, OSKAR_SKY_LINE_WIDTH_HZ, 0, 1));
        oskar_sky_free(sky, &status);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (ra,dec,i) format\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "Format RA, Dec, I\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (ra,dec,i) = format blah\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, Dec) = format\n");
        (void) fprintf(file, "1, 2\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, I) = format\n");
        (void) fprintf(file, "1, 2\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (Ra, RaD, I) = format\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        (void) fprintf(file, "# (I) = format\n");
        (void) fprintf(file, "1\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_INVALID_ARGUMENT, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
    {
        int status = 0;
        FILE* file = fopen(filename, "w");
        // Don't include the word "format" in the header, otherwise it will
        // assume it's malformed and return an invalid argument error.
        (void) fprintf(file, "# no special header\n");
        (void) fprintf(file, "1, 2, 3\n");
        (void) fclose(file);
        oskar_Sky* sky = oskar_sky_load_named_columns(filename, type, &status);
        ASSERT_EQ((int) OSKAR_ERR_FILE_IO, status);
        ASSERT_EQ(0, sky);
        (void) remove(filename);
    }
}
