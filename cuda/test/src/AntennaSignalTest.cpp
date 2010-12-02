#include "cuda/test/AntennaSignalTest.h"
#include "cuda/antennaSignal2dHorizontalIsotropic.h"
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
CPPUNIT_TEST_SUITE_REGISTRATION(AntennaSignalTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void AntennaSignalTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void AntennaSignalTest::tearDown()
{
}

/**
 * @details
 * Tests antenna signal generation using CUDA.
 */
void AntennaSignalTest::test_method()
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

    // Generate some source positions.
    float centreAz = 0;  // Beam azimuth.
    float centreEl = 50; // Beam elevation.
    SphericalPositions<float> pos (
            centreAz * DEG2RAD, centreEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            1 * DEG2RAD, 1 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Generate source amplitudes.
    std::vector<float> samp(ns, 1.0);

    // Call CUDA antenna signal generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> signals(na*na * 2); // Antenna signal real & imaginary values.
    TIMER_START
    antennaSignal2dHorizontalIsotropic(na*na, &ax[0], &ay[0], ns, &samp[0],
            &slon[0], &slat[0], 2 * M_PI * (freq / C_0), &signals[0]);
    TIMER_STOP("Finished antenna signal generation "
            "(%d antennas, %d sources)", na*na, ns);

    // Write signals data to file.
    FILE* file = fopen("antennaSignal2dHorizontalIsotropic.dat", "w");
    for (unsigned a = 0; a < na*na; ++a) {
        fprintf(file, "%10d%16.4e%16.4e\n", a, signals[2*a], signals[2*a+1]);
    }
    fclose(file);
}
