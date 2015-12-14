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
#include <oskar_settings_types.hpp>

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

    bool ok = true;
    DateTime t;
    // British style
    {
        t.fromString("1-2-2015 10:05:23.2", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("01-02-2015 10:05:23.2", t.toString().c_str());
    }
    // CASA style
    {
        t.fromString("2015/1/2/03:04:05.6", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("2015/01/02/03:04:05.6", t.toString().c_str());
    }
    // International style
    {
        t.fromString("2015-2-3 04:05:06.7", &ok);
        ASSERT_TRUE(ok);
    }
    // ISO style
    {
        t.fromString("2015-3-4T05:06:07.8910111213", &ok);
        ASSERT_TRUE(ok);
    }
    // MJD
    {
        // TODO
    }
}

