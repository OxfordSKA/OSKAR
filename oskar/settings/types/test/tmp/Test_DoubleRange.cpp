/*
 * Copyright (c) 2013, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <climits>
#include <cfloat>
#include "../../oskar_DoubleRange.hpp"

using namespace oskar;
using namespace std;

// needs to be able to store "min" 0->DBL_MAX (flux range min)
// needs to be able to store "max" 0->DBL_MAX (flux range max)

TEST(DoubleRange, test1)
{
    DoubleRangeExt r;
    r.set("1.2345e100");
    r.set("hello");
}

TEST(DoubleRange, test2)
{
    DoubleRangeExt r(0.0, DBL_MAX, "min");
    bool ok = false;
    ASSERT_DOUBLE_EQ(0.0, r.getDouble(&ok));
    ASSERT_FALSE(ok);
}

TEST(DoubleRange, test3)
{
    // Use case range set from 0.0 to 100.0 and initial value set to "min"
    DoubleRangeExt r(0.0, 100.0, "min");
    ASSERT_DOUBLE_EQ(0.0, r.min());
    ASSERT_DOUBLE_EQ(100.0, r.max());

    bool ok = false;
    ASSERT_DOUBLE_EQ(0.0, r.getDouble(&ok));
    ASSERT_FALSE(ok);
    ASSERT_STREQ(std::string().c_str(), r.toString(&ok).c_str());
    ASSERT_FALSE(ok);

#if 0
    // Try to set a double value of -1.0, this should result in the string
    // remaining at min.
    r.set(-1.0);
    ASSERT_DOUBLE_EQ(0.0-DBL_MIN, r.getDouble(&ok));
    ASSERT_FALSE(ok);
    ASSERT_STREQ("min", r.toString(&ok).c_str());
    ASSERT_TRUE(ok);

    // Try to set a double value of 101, this should result in the value
    // becoming double 100.0 and the text being "100"
    r.set(101.0, &ok);
    ASSERT_FALSE(ok);
    ASSERT_DOUBLE_EQ(100.0, r.getDouble(&ok));
    ASSERT_TRUE(ok);
    ASSERT_STREQ("100", r.toString(&ok).substr(0,3).c_str());
    ASSERT_TRUE(ok);

    // Try to set a double value of 49.5678e3, this should result in the value
    // becoming double 49.56 and the text being "49.56"
    r.set(49.56);
    ASSERT_DOUBLE_EQ(49.56, r.getDouble(&ok));
    ASSERT_TRUE(ok);
    EXPECT_STREQ("49.56", r.toString(&ok).substr(0, 5).c_str());
    ASSERT_TRUE(ok);
#endif
}
