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

#ifndef TEST_TELESCOPE_MODEL_LOAD_SAVE_H_
#define TEST_TELESCOPE_MODEL_LOAD_SAVE_H_

/**
 * @file Test_telescope_model_load_save.h
 */

#include <cppunit/extensions/HelperMacros.h>
#include <QtCore/QtCore>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class Test_telescope_model_load_save : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(Test_telescope_model_load_save);
        CPPUNIT_TEST(test_0_level);
        CPPUNIT_TEST(test_1_level);
        CPPUNIT_TEST(test_2_level);
        CPPUNIT_TEST(test_load_telescope_noise_stddev);
        CPPUNIT_TEST(test_load_telescope_noise_sensitivity);
        CPPUNIT_TEST(test_load_telescope_noise_t_sys);
        CPPUNIT_TEST(test_load_telescope_noise_t_components);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Test method.
        void test_0_level();
        void test_1_level();
        void test_2_level();

        void test_load_telescope_noise_stddev();
        void test_load_telescope_noise_sensitivity();
        void test_load_telescope_noise_t_sys();
        void test_load_telescope_noise_t_components();

    private:
        void generate_noisy_telescope(const QString& dir,
                int num_stations, int write_depth, const QVector<double>& freqs,
                const QHash< QString, QVector<double> >& noise);
};

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(Test_telescope_model_load_save);

#endif // TEST_TELESCOPE_MODEL_LOAD_SAVE_H_
