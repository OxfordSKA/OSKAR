#ifndef BEAM_PATTERN_RANDOM_TEST_H
#define BEAM_PATTERN_RANDOM_TEST_H

/**
 * @file BeamPatternRandomTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class BeamPatternRandomTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(BeamPatternRandomTest);
        CPPUNIT_TEST(test);
        CPPUNIT_TEST(test_scattered);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test method.
        void test();

        /// Test method.
        void test_scattered();
};

#endif // BEAM_PATTERN_RANDOM_TEST_H
