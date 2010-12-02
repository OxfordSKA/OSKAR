#ifndef ANTENNA_SIGNAL_TEST_H
#define ANTENNA_SIGNAL_TEST_H

/**
 * @file AntennaSignalTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class AntennaSignalTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(AntennaSignalTest);
        CPPUNIT_TEST(test_method);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test method.
        void test_method();
};

#endif // ANTENNA_SIGNAL_TEST_H
