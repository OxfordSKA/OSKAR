#ifndef SPHERICAL_POSITIONS_TEST_H
#define SPHERICAL_POSITIONS_TEST_H

/**
 * @file SphericalPositionsTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class SphericalPositionsTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(SphericalPositionsTest);
        CPPUNIT_TEST(test_create);
        CPPUNIT_TEST(test_generate13x13);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test creation.
        void test_create();

        /// Test generation of points.
        void test_generate13x13();
};

#endif // SPHERICAL_POSITIONS_TEST_H
