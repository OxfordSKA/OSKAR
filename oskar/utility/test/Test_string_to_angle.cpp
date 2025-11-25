/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "utility/oskar_string_to_angle.h"

#define DEG2RAD (M_PI / 180.0)


TEST(string_to_angle, hours)
{
    {
        int status = 0;
        const char* in = "123.456rad";
        const double x = 123.456;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "123.456deg";
        const double x = 123.456 * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "-00:01:01.1";
        const double x = -15 * (1.0 / 60.0 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "-00h01m01.1";
        const double x = -15 * (1.0 / 60 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "   -00h01m01.1   ";
        const double x = -15 * (1.0 / 60 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "3.14";
        const double x = 3.14;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "3.14";
        const double x = 3.14 * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'd', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "11:59:59.123";
        const double x = 15 * (11 + 59.0 / 60.0 + 59.123 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_hours_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "11:59:61.123";
        (void) oskar_string_hours_to_radians(in, 'r', &status);
        ASSERT_EQ((int) OSKAR_ERR_BAD_UNITS, status);
    }
}


TEST(string_to_angle, degrees)
{
    {
        int status = 0;
        const char* in = "123.456rad";
        const double x = 123.456;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "123.456deg";
        const double x = 123.456 * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "-00.01.01.1";
        const double x = -(1.0 / 60.0 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "     -00.01.01.1  ";
        const double x = -(1.0 / 60.0 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "-00d01m01.1";
        const double x = -(1.0 / 60 + 1.1 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "3.14";
        const double x = 3.14;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "3.14";
        const double x = 3.14 * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'd', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "11.59.59.123";
        const double x = (11 + 59.0 / 60.0 + 59.123 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "11:59:59.123";
        const double x = (11 + 59.0 / 60.0 + 59.123 / 3600.0) * DEG2RAD;
        ASSERT_DOUBLE_EQ(x, oskar_string_degrees_to_radians(in, 'r', &status));
        ASSERT_EQ(0, status);
    }
    {
        int status = 0;
        const char* in = "11.59.61.123";
        (void) oskar_string_degrees_to_radians(in, 'r', &status);
        ASSERT_EQ((int) OSKAR_ERR_BAD_UNITS, status);
    }
}
