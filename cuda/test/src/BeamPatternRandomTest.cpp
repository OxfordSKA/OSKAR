#include "cuda/test/BeamPatternRandomTest.h"
#include "cuda/beamPattern2dHorizontalGeometric.h"
#include "cuda/beamPattern2dHorizontalWeights.h"
#include "math/core/SphericalPositions.h"
#include "math/core/GridPositions.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define C_0 299792458.0

#define TIMER_ENABLE 1
#include "utility/timer.h"

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(BeamPatternRandomTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void BeamPatternRandomTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void BeamPatternRandomTest::tearDown()
{
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void BeamPatternRandomTest::test()
{
    // Generate array of antenna positions.
    int seed = 10;
    float radius = 15; // metres. (was 125)
    float xs = 1.4, ys = 1.4, xe = 0.2, ye = 0.2; // separations, errors.

    int na = GridPositions::circular<float>(seed, radius, xs, ys, xe, ye, 0, 0);
    std::vector<float> ax(na), ay(na); // Antenna (x,y) positions.
    GridPositions::circular(seed, radius, xs, ys, xe, ye, &ax[0], &ay[0]);

    // Write antenna positions to file.
    FILE* file = fopen("arrayRandom.dat", "w");
    for (int a = 0; a < na; ++a) {
        fprintf(file, "%12.3f%12.3f\n", ax[a], ay[a]);
    }
    fclose(file);

    // Set beam direction.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 50; // Beam elevation.

    // Generate test source positions for a square image.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.

    // Generate test source positions for the hemisphere.
//    SphericalPositions<float> pos (
//            180 * DEG2RAD, 45 * DEG2RAD, // Centre.
//            180 * DEG2RAD, 45 * DEG2RAD, // Half-widths.
//            0.03 * DEG2RAD, 0.03 * DEG2RAD, // Spacings.
//            0.0, true, false, true, true,
//            SphericalPositions<float>::PROJECTION_NONE);

    int ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.

    // Call CUDA beam pattern generator.
    int nf = 7; // Number of frequencies.
    float freq[] = {70, 115, 150, 200, 240, 300, 450}; // Frequencies in MHz.
    for (int f = 0; f < nf; ++f) {
        TIMER_START
        beamPattern2dHorizontalWeights(na, &ax[0], &ay[0], ns, &slon[0],
                &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
                2 * M_PI * (freq[f] * 1e6 / C_0), &image[0]);
        TIMER_STOP("Finished beam pattern (%.0f MHz)", freq[f]);

        // Write image data to file.
        char fname[200];
        snprintf(fname, 200, "beamPattern_%.0f.dat", freq[f]);
        file = fopen(fname, "w");
        for (int s = 0; s < ns; ++s) {
            fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                    slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
        }
        fclose(file);
    }
}
