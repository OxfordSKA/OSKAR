#ifndef BEAMFORMER_MATRIX_VECTOR_TEST_H
#define BEAMFORMER_MATRIX_VECTOR_TEST_H

/**
 * @file BeamformerMatrixVectorTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class BeamformerMatrixVectorTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(BeamformerMatrixVectorTest);
        CPPUNIT_TEST(test_basicMatrixVector);
        CPPUNIT_TEST(test_method);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Set up context before running a test.
        void setUp();

        /// Clean up after the test run.
        void tearDown();

    public:
        /// Test basic complex matrix-vector multiplication.
        void test_basicMatrixVector();

        /// Test method.
        void test_method();
};

#endif // BEAMFORMER_MATRIX_VECTOR_TEST_H
