#ifndef BEAM_PATTERN_TEST_H
#define BEAM_PATTERN_TEST_H

/**
 * @file BeamPatternTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class BeamPatternTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(BeamPatternTest);
        CPPUNIT_TEST(test_direct);
        CPPUNIT_TEST(test_weights);
        CPPUNIT_TEST(test_weights_random);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test method.
        void test_direct();

        /// Test method.
        void test_weights();

        /// Test method.
        void test_weights_random();
};

#endif // BEAM_PATTERN_TEST_H
