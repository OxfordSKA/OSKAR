#include "cuda/test/BeamPatternTest.h"
#include "cuda/beamPatternDirect.h"
#include "cuda/beamPatternWeights.h"
#include "math/core/SphericalPositions.h"
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
CPPUNIT_TEST_SUITE_REGISTRATION(BeamPatternTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void BeamPatternTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void BeamPatternTest::tearDown()
{
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void BeamPatternTest::test_direct()
{
    // Generate square array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na * na), ay(na * na); // Antenna (x,y) positions.
    for (int x = 0; x < na; ++x) {
        for (int y = 0; y < na; ++y) {
            int i = y + x * na;
            ax[i] = x * sep - halfArraySize;
            ay[i] = y * sep - halfArraySize;
        }
    }

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 50; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    beamPatternDirect(na*na, &ax[0], &ay[0], ns, &slon[0], &slat[0],
            beamAz * DEG2RAD, beamEl * DEG2RAD, 2 * M_PI * (freq / C_0),
            &image[0]);
    TIMER_STOP("Finished beam pattern (direct)");

    // Write image data to file.
    FILE* file = fopen("beamPatternDirect.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void BeamPatternTest::test_weights()
{
    // Generate square array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na * na), ay(na * na); // Antenna (x,y) positions.
    for (int x = 0; x < na; ++x) {
        for (int y = 0; y < na; ++y) {
            int i = y + x * na;
            ax[i] = x * sep - halfArraySize;
            ay[i] = y * sep - halfArraySize;
        }
    }

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 50; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    beamPatternWeights(na*na, &ax[0], &ay[0], ns, &slon[0], &slat[0],
            beamAz * DEG2RAD, beamEl * DEG2RAD, 2 * M_PI * (freq / C_0),
            &image[0]);
    TIMER_STOP("Finished beam pattern (separate weights)");

    // Write image data to file.
    FILE* file = fopen("beamPatternWeights.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}
