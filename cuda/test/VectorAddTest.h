#ifndef VECTOR_ADD_TEST_H
#define VECTOR_ADD_TEST_H

/**
 * @file VectorAddTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

/**
 * @class
 */

class VectorAddTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( VectorAddTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        VectorAddTest();
        ~VectorAddTest();

    private:
};

#endif // VECTOR_ADD_TEST_H
