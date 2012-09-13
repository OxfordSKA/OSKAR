/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef TEST_SKY_MODEL_H_
#define TEST_SKY_MODEL_H_

/**
 * @file Test_SkyModel.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class Test_SkyModel : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(Test_SkyModel);
        CPPUNIT_TEST(test_resize);
        CPPUNIT_TEST(test_set_source);
        CPPUNIT_TEST(test_append);
        CPPUNIT_TEST(test_load);
        CPPUNIT_TEST(test_compute_relative_lmn);
        CPPUNIT_TEST(test_horizon_clip);
        CPPUNIT_TEST(test_split);
        CPPUNIT_TEST(test_filter_by_radius);
        CPPUNIT_TEST(test_gaussian_source);
        CPPUNIT_TEST(test_evaluate_gaussian_source_parameters);
        CPPUNIT_TEST(test_insert);
        CPPUNIT_TEST(test_sky_model_set);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Test method.
        void test_resize();
        void test_set_source();
        void test_append();
        void test_load();
        void test_compute_relative_lmn();
        void test_horizon_clip();
        void test_split();
        void test_filter_by_radius();
        void test_gaussian_source();
        void test_evaluate_gaussian_source_parameters();
        void test_insert();
        void test_sky_model_set();
};

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(Test_SkyModel);

#endif // TEST_SKY_MODEL_H_
