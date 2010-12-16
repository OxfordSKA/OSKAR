#ifndef MATRIX3_TEST_H
#define MATRIX3_TEST_H

/**
 * @file Matrix3Test.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class Matrix3Test : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(Matrix3Test);
        CPPUNIT_TEST(test_identity);
        CPPUNIT_TEST(test_multiply);
        CPPUNIT_TEST(test_rotation);
        CPPUNIT_TEST(test_scale);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test identity.
        void test_identity();

        /// Test multiply.
        void test_multiply();

        /// Test rotation.
        void test_rotation();

        /// Test scale.
        void test_scale();
};

#endif // MATRIX3_TEST_H
