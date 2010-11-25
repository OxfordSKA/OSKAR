#include "math/core/test/SphericalPositionsTest.h"
#include "math/core/SphericalPositions.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(SphericalPositionsTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void SphericalPositionsTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void SphericalPositionsTest::tearDown()
{
}

/**
 * @details
 * Tests object creation.
 */
void SphericalPositionsTest::test_create()
{
    SphericalPositions<float> posGen(0, 90 * DEG2RAD,
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths
            0.5 * DEG2RAD, 0.5 * DEG2RAD);
}

/**
 * @details
 * Tests generation of points using parameters to generate a 13 x 13 grid.
 */
void SphericalPositionsTest::test_generate13x13()
{
    SphericalPositions<float> posGen(0, 90 * DEG2RAD,
            2 * DEG2RAD, 2 * DEG2RAD, // Half-widths
            0.3 * DEG2RAD, 0.3 * DEG2RAD);

    // Should have a 13 x 13 point grid.
    unsigned points = posGen.generate(0 ,0);
    CPPUNIT_ASSERT_EQUAL(169u, points);
    std::vector<float> longitudes(points);
    std::vector<float> latitudes(points);
    posGen.generate(&longitudes[0], &latitudes[0]);
}
