#include "cuda/test/VectorAddTest.h"
#include "cuda/vectorAdd.h"

CPPUNIT_TEST_SUITE_REGISTRATION( VectorAddTest );

/**
 *@details DataTypesTest
 */
VectorAddTest::VectorAddTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
VectorAddTest::~VectorAddTest()
{
}

void VectorAddTest::setUp()
{
}

void VectorAddTest::tearDown()
{
}

void VectorAddTest::test_method()
{
    vectorAdd();
}
