#include "cuda/test/BeamPatternRandomTest.h"
#include "cuda/beamPattern2dHorizontalWeights.h"
#include "math/core/SphericalPositions.h"
#include "math/core/GridPositions.h"
#include "math/core/Matrix3.h"
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
    float xs = 1.4, ys = 1.4, xe = 0.3, ye = 0.3; // separations, errors.

    int na = GridPositions::circular(seed, radius, xs, ys, xe, ye);
    std::vector<float> ax(na), ay(na), az(na); // Antenna positions.
    GridPositions::circular(seed, radius, xs, ys, xe, ye, &ax[0], &ay[0]);

    // Rotate around z.
    float matrix[9];
    Matrix3::rotationZ(matrix, float(0 * DEG2RAD));
    Matrix3::transformPoints(matrix, na, &ax[0], &ay[0], &az[0]);

    // Write antenna positions to file.
    FILE* file = fopen("arrayRandom.dat", "w");
    for (int a = 0; a < na; ++a) {
        fprintf(file, "%12.3f%12.3f\n", ax[a], ay[a]);
    }
    fclose(file);

    // Set beam direction.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 90; // Beam elevation.

    // Generate test source positions for a square image.
//    SphericalPositions<float> pos (
//            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
//            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
//            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.

    // Generate test source positions for the hemisphere.
    SphericalPositions<float> pos (
            180 * DEG2RAD, 45 * DEG2RAD, // Centre.
            180 * DEG2RAD, 45 * DEG2RAD, // Half-widths.
            //0.03 * DEG2RAD, 0.03 * DEG2RAD, // Spacings.
            0.2 * DEG2RAD, 0.2 * DEG2RAD, // Spacings.
            0.0, true, false, true, true,
            SphericalPositions<float>::PROJECTION_NONE);

    int ns = 1 + pos.generate(0, 0); // No. of sources (add a point at zenith).
    std::vector<float> slon(ns), slat(ns);
    slon[0] = 0; slat[0] = 90 * DEG2RAD; // Add a point at zenith.
    pos.generate(&slon[1], &slat[1]); // Add a point at zenith.
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

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void BeamPatternRandomTest::test_scattered()
{
    // Generate array of antenna positions.
    int seed = 10;
    float radius = 15; // metres. (was 125)
    float xs = 1.4, ys = 1.4, xe = 0.3, ye = 0.3; // separations, errors.

    int na = GridPositions::circular(seed, radius, xs, ys, xe, ye);
    std::vector<float> ax(na), ay(na), az(na); // Antenna positions.
    GridPositions::circular(seed, radius, xs, ys, xe, ye, &ax[0], &ay[0]);

    // Rotate around z.
    float matrix[9];
    Matrix3::rotationZ(matrix, float(0 * DEG2RAD));
    Matrix3::transformPoints(matrix, na, &ax[0], &ay[0], &az[0]);

    // Write antenna positions to file.
    FILE* file = fopen("arrayRandomScattered.dat", "w");
    for (int a = 0; a < na; ++a) {
        fprintf(file, "%12.3f%12.3f\n", ax[a], ay[a]);
    }
    fclose(file);

    // Set beam direction.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 90; // Beam elevation.

//    SphericalPositions<float> pos (
//            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
//            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
//            0.5 * DEG2RAD, 0.5 * DEG2RAD); // Spacings.
//    int ns = pos.generate(0, 0); // No. of sources.
//    std::vector<float> slon(ns), slat(ns);
//    pos.generate(&slon[0], &slat[0]);

    float slat[] = {1, 0.75, 1.3, 0.8,
            1.5708, 1.5621, 1.5533, 1.5446, 1.5359,
            1.5272, 1.5184, 1.5097, 1.5010};
    float slon[] = {-2.8304, -0.0655, -1.9320, -2.3682,
            0, 0, 0, 0, 0,
            0, 0, 0, 0};
    int ns = sizeof(slon) / sizeof(float);
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
        snprintf(fname, 200, "beamPatternScattered_%.0f.dat", freq[f]);
        file = fopen(fname, "w");
        for (int s = 0; s < ns; ++s) {
            fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                    slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
        }
        fclose(file);
    }
}
