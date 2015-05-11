/*
 * Copyright (c) 2015, The University of Oxford
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
#include <oskar_settings_types.hpp>

using namespace oskar;

TEST(settings_types, DoubleRange)
{

    DoubleRange r;
//    {
//        ASSERT_STREQ("0", r.toString().c_str());
//    }
//    {
//        bool ok = true;
//        r.init("2.0,5.0", &ok);
//        ASSERT_TRUE(ok);
//        ASSERT_STREQ("0", r.toString().c_str());
//    }
//    {
//        bool ok = true;
//        r.fromString("2.12345678", &ok);
//        ASSERT_TRUE(ok);
//        ASSERT_STREQ("2.12345678", r.toString().c_str());
//        ASSERT_STREQ("2.12345678", r.toString("%.8f").c_str());
//        ASSERT_STREQ("2.1235", r.toString("%.4f").c_str());
//        ASSERT_STREQ("2.123457e+00", r.toString("%e").c_str());
//        ASSERT_STREQ("0002.1", r.toString("%06.1f").c_str());
//    }
//    {
//        bool ok = true;
//        r.fromString("1.0", &ok);
//        ASSERT_FALSE(ok);
//        ASSERT_STREQ("2", r.toString().c_str());
//    }
//
//    {
//        bool ok = true;
//        r.fromString("10.0", &ok);
//        ASSERT_FALSE(ok);
//        ASSERT_STREQ("5", r.toString().c_str());
//    }
    {
        bool ok = true;
        r.init("MIN,MAX", &ok);
        ASSERT_TRUE(ok);
        r.fromString("0.999999999999", &ok);
        ASSERT_TRUE(ok);
        //std::cout << r.toString() << std::endl;
    }
}

