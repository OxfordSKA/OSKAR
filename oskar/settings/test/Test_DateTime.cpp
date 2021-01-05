/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"
#include <cstdio>

using namespace std;
using namespace oskar;

TEST(settings_types, DateTime)
{
    /*
     *  d-M-yyyy h:m:s[.z] - British style
     *  yyyy/M/d/h:m:s[.z] - CASA style
     *  yyyy-M-d h:m:s[.z] - International style
     *  yyyy-M-dTh:m:s[.z] - ISO date style
     *  MJD
     */
    DateTime t;
    // British style
    {
        EXPECT_TRUE(t.set_default("1-2-2015 10:05:23.2"));
        EXPECT_TRUE(t.is_default());
        EXPECT_STREQ("01-02-2015 10:05:23.2", t.get_value());
    }
    // CASA style
    {
        EXPECT_TRUE(t.set_value("2015/1/2/03:04:05.6"));
        EXPECT_FALSE(t.is_default());
        EXPECT_STREQ("2015/01/02/03:04:05.6", t.get_value());
    }
    // International style
    {
        EXPECT_TRUE(t.set_value("2015-2-3 04:05:06.7"));
        EXPECT_FALSE(t.is_default());
        EXPECT_DOUBLE_EQ(6.7, t.value().seconds);
        EXPECT_DOUBLE_EQ(23.2, t.default_value().seconds);
        EXPECT_STREQ("2015-02-03 04:05:06.7", t.get_value());
    }
    // International style
    {
        EXPECT_TRUE(t.set_value("2015-12-31 23:59:59.0"));
        EXPECT_FALSE(t.is_default());
        EXPECT_DOUBLE_EQ(59.0, t.value().seconds);
        EXPECT_EQ(59, t.value().minutes);
        EXPECT_EQ(23, t.value().hours);
        EXPECT_EQ(31, t.value().day);
        EXPECT_EQ(12, t.value().month);
        EXPECT_EQ(2015, t.value().year);
        EXPECT_DOUBLE_EQ(23.2, t.default_value().seconds);
        EXPECT_STREQ("2015-12-31 23:59:59.0", t.get_value());
    }
    // ISO style
    {
        EXPECT_TRUE(t.set_value("2015-3-4T05:06:07.8910111213"));
        EXPECT_FALSE(t.is_default());
    }
    // MJD1
    {
        double mjd = t.to_mjd();
        double mjd2 = t.to_mjd_2();
        EXPECT_DOUBLE_EQ(mjd, mjd2);
        t.from_mjd(mjd);
        EXPECT_DOUBLE_EQ(mjd, t.to_mjd());

    }
    // MJD2
    {
        double mjd = 46113.7511111;
        t.from_mjd(mjd);
        EXPECT_DOUBLE_EQ(mjd, t.to_mjd());
        EXPECT_EQ(DateTime::MJD, t.format());
    }
    {
        DateTime t1;
        EXPECT_TRUE(t1.set_value("46113.7654321"));
        EXPECT_EQ(DateTime::MJD, t1.format());
        EXPECT_STREQ("46113.7654321", t1.get_value());
        EXPECT_DOUBLE_EQ(46113.7654321, t1.to_mjd());
    }
    {
        DateTime t1;
        t1.set_value("45464844.54646541");
        EXPECT_STREQ("45464844.546465", t1.get_value());
    }

    // Failure modes.
    // British style
    {
        EXPECT_FALSE(t.set_value("01-13-2015 10:05:23.2"));
        EXPECT_FALSE(t.set_value("00-12-2015 10:05:23.2"));
        EXPECT_FALSE(t.set_value("01-12-2015 25:05:23.2"));
        EXPECT_FALSE(t.set_value("01-12-2015 22:60:23.2"));
        EXPECT_FALSE(t.set_value("01-12-2015 22:59:61.2"));
    }
    // CASA style
    {
        EXPECT_FALSE(t.set_value("2015/13/1/03:04:05.6"));
        EXPECT_FALSE(t.set_value("2015/12/0/03:04:05.6"));
        EXPECT_FALSE(t.set_value("2015/12/1/25:04:05.6"));
        EXPECT_FALSE(t.set_value("2015/12/1/22:60:05.6"));
        EXPECT_FALSE(t.set_value("2015/12/1/22:59:61.6"));
    }
    // International style
    {
        EXPECT_FALSE(t.set_value("2015-13-1 04:05:06.7"));
        EXPECT_FALSE(t.set_value("2015-12-33 04:05:06.7"));
        EXPECT_FALSE(t.set_value("2015-12-0 04:05:06.7"));
        EXPECT_FALSE(t.set_value("2015-12-1 25:05:06.7"));
        EXPECT_FALSE(t.set_value("2015-12-1 22:60:06.7"));
        EXPECT_FALSE(t.set_value("2015-12-1 22:59:61.7"));
    }
    // ISO style
    {
        EXPECT_FALSE(t.set_value("2015-13-4T05:06:07.8910111213"));
        EXPECT_FALSE(t.set_value("2015-12-0T05:06:07.8910111213"));
        EXPECT_FALSE(t.set_value("2015-12-1T25:06:07.8910111213"));
        EXPECT_FALSE(t.set_value("2015-12-1T22:60:07.8910111213"));
        EXPECT_FALSE(t.set_value("2015-12-1T22:59:67.8910111213"));
    }

    // Comparisons.
    {
        DateTime t1, t2;
        EXPECT_TRUE(t1.set_value("1-2-2015 10:05:23.2"));
        EXPECT_TRUE(t2.set_value("1-2-2015 10:05:23.2"));
        EXPECT_TRUE(t1 == t2);
        EXPECT_TRUE(t1.set_value("1-2-2015 10:05:23.2"));
        EXPECT_TRUE(t2.set_value("1-2-2015 10:06:23.2"));
        EXPECT_TRUE(t2 > t1);
    }
}
