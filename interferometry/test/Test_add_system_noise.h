/*
 * Copyright (c) 2012, The University of Oxford
 * All rights reserved.
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

#ifndef TEST_ADD_SYSTEM_NOISE_H_
#define TEST_ADD_SYSTEM_NOISE_H_

/**
 * @file Test_add_system_noise.h
 */

#include <cppunit/extensions/HelperMacros.h>

#include "utility/oskar_Mem.h"
#include "imaging/oskar_Image.h"
#include "interferometry/oskar_TelescopeModel.h"

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class Test_add_system_noise : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(Test_add_system_noise);
        CPPUNIT_TEST(test_stddev);
        CPPUNIT_TEST_SUITE_END();

    public:
        // Test Methods
        void test_stddev();

    private:
        void generate_range(oskar_Mem* data, int number, double start, double inc);
        void check_image_stats(oskar_Image* image, oskar_TelescopeModel* tel);
};

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(Test_add_system_noise);

#endif /* TEST_ADD_SYSTEM_NOISE_H_ */
