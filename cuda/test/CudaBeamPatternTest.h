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

#ifndef CUDA_BEAM_PATTERN_TEST_H_
#define CUDA_BEAM_PATTERN_TEST_H_

/**
 * @file CudaBeamPatternTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class CudaBeamPatternTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(CudaBeamPatternTest);
//        CPPUNIT_TEST(test_regular);
//        CPPUNIT_TEST(test_superStation);
//        CPPUNIT_TEST(test_satStation);
//        CPPUNIT_TEST(test_stations200);
//        CPPUNIT_TEST(test_stations2000);
//        CPPUNIT_TEST(test_stations4000);
//        CPPUNIT_TEST(test_random);
//        CPPUNIT_TEST(test_perturbed);
//        CPPUNIT_TEST(test_scattered);
        CPPUNIT_TEST(test_single_precision);
        CPPUNIT_TEST(test_double_precision);
        CPPUNIT_TEST(test_single_precision_precomputed);
        CPPUNIT_TEST(test_double_precision_precomputed);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test method.
        void test_regular();

        /// Test method.
        void test_superStation();

        /// Test method.
        void test_satStation();

        /// Test method.
        void test_stations200();

        /// Test method.
        void test_stations2000();

        /// Test method.
        void test_stations4000();

        /// Test method.
        void test_perturbed();

        /// Test method.
        void test_scattered();

        /// Test method.
        void test_random();

        /// Test method.
        void test_single_precision();

        /// Test method.
        void test_double_precision();

        /// Test method.
        void test_single_precision_precomputed();

        /// Test method.
        void test_double_precision_precomputed();
};

#endif // CUDA_BEAM_PATTERN_TEST_H_
