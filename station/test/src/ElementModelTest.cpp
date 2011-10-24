/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "station/test/ElementModelTest.h"
#include "station/oskar_element_model_load.h"
#include "station/oskar_ElementModel.h"

#include <cmath>

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Sets up the context before running each test method.
 */
void ElementModelTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void ElementModelTest::tearDown()
{
}

/**
 * @details
 * Tests loading of antenna pattern data.
 */
void ElementModelTest::test_method()
{
    // Create a dummy antenna pattern.
    char data[] = ""
            "Theta [deg.]  Phi   [deg.]  Abs(Dir.)[dBi   ]   Abs(Theta)[dBi   ]  Phase(Theta)[deg.]  Abs(Phi  )[dBi   ]  Phase(Phi  )[deg.]  Ax.Ratio[dB    ]    \n"
            "------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            "   0.000           0.000           8.519e+000         -2.854e+000             316.678          8.190e+000              12.037         -1.003e+001     \n"
            "  10.000           0.000           8.034e+000         -2.978e+000             324.125          7.675e+000              14.225         -9.724e+000     \n"
            "  20.000           0.000           6.526e+000         -2.576e+000             346.665          5.957e+000              21.366         -8.752e+000     \n"
            "  30.000           0.000           4.086e+000         -5.775e-001              16.161          2.270e+000              36.255         -7.478e+000     \n"
            "  40.000           0.000           2.590e+000          1.732e+000              41.065         -4.875e+000              77.772         -1.070e+001     \n"
            "  50.000           0.000           3.440e+000          2.657e+000              62.472         -4.383e+000             173.916         -1.506e+001     \n"
            "  60.000           0.000           3.724e+000          1.500e+000              84.882         -2.463e-001             210.558         -1.691e+001     \n"
            "  70.000           0.000           1.969e+000         -2.060e+000             114.111         -2.162e-001             232.236         -1.815e+001     \n"
            "  80.000           0.000          -2.457e+000         -6.794e+000             161.124         -4.453e+000             251.156         -2.066e+001     \n"
            "  90.000           0.000          -8.131e+000         -8.132e+000             202.659         -4.991e+001             269.010          1.700e+001     \n"
            " 100.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 110.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 120.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 130.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 140.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 150.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 160.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 170.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 180.000           0.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            "   0.000          10.000           8.519e+000          2.706e-001             337.613          7.815e+000              14.447         -1.003e+001     \n"
            "  10.000          10.000           8.023e+000          9.926e-002             342.041          7.259e+000              16.541         -9.776e+000     \n"
            "  20.000          10.000           6.461e+000         -6.502e-002             355.956          5.368e+000              23.452         -9.005e+000     \n"
            "  30.000          10.000           3.771e+000          4.122e-001              17.940          1.084e+000              38.395         -8.096e+000     \n"
            "  40.000          10.000           1.807e+000          1.347e+000              41.633         -8.167e+000              95.058         -1.122e+001     \n"
            "  50.000          10.000           2.988e+000          1.573e+000              63.931         -2.572e+000             196.071         -1.460e+001     \n"
            "  60.000          10.000           3.880e+000          1.002e-001              86.938          1.524e+000             223.119         -1.493e+001     \n"
            "  70.000          10.000           2.609e+000         -3.684e+000             116.173          1.447e+000             242.561         -1.491e+001     \n"
            "  80.000          10.000          -1.841e+000         -8.804e+000             163.911         -2.817e+000             260.783         -1.691e+001     \n"
            "  90.000          10.000          -1.026e+001         -1.026e+001             207.361         -4.829e+001             278.424          1.552e+001     \n"
            " 100.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 110.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 120.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 130.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 140.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 150.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 160.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 170.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 180.000          10.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            "   0.000          20.000           8.519e+000          2.808e+000             348.289          7.161e+000              17.146         -1.003e+001     \n"
            "  10.000          20.000           8.039e+000          2.501e+000             351.168          6.616e+000              19.120         -9.867e+000     \n"
            "  20.000          20.000           6.512e+000          1.762e+000               0.507          4.741e+000              25.630         -9.406e+000     \n"
            "  30.000          20.000           3.686e+000          9.652e-001              17.229          3.646e-001              39.509         -8.860e+000     \n"
            "  40.000          20.000           7.556e-001          3.991e-001              39.428         -1.028e+001              99.051         -1.084e+001     \n"
            "  50.000          20.000           1.694e+000         -3.334e-001              63.289         -2.589e+000             206.009         -1.320e+001     \n"
            "  60.000          20.000           3.239e+000         -2.127e+000              87.999          1.747e+000             230.445         -1.260e+001     \n"
            "  70.000          20.000           2.509e+000         -5.694e+000             117.367          1.797e+000             249.197         -1.222e+001     \n"
            "  80.000          20.000          -1.732e+000         -1.030e+001             159.607         -2.382e+000             267.197         -1.451e+001     \n"
            "  90.000          20.000          -1.205e+001         -1.205e+001             197.312         -4.783e+001             284.795          1.278e+001     \n"
            " 100.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 110.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 120.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 130.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 140.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 150.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 160.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 170.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n"
            " 180.000          20.000          -2.000e+002         -2.000e+002             134.449         -2.000e+002             134.449          1.700e+001     \n";

    // Write it to file.
    char filename[] = "dummy_pattern.dat";
    FILE* file = fopen(filename, "w");
    fwrite(data, sizeof(char), sizeof(data)/sizeof(char), file);
    fclose(file);

    // Load the file.
    oskar_ElementModel pattern;
    oskar_element_model_load(filename, &pattern);

    // Check the contents of the data.
    CPPUNIT_ASSERT_EQUAL(30, pattern.n_points);
    CPPUNIT_ASSERT_EQUAL(10, pattern.n_theta);
    CPPUNIT_ASSERT_EQUAL(3, pattern.n_phi);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0 * M_PI / 180.0, pattern.min_theta, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0 * M_PI / 180.0, pattern.min_phi, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(90.0 * M_PI / 180.0, pattern.max_theta, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(20.0 * M_PI / 180.0, pattern.max_phi, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0 * M_PI / 180.0, pattern.inc_theta, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0 * M_PI / 180.0, pattern.inc_phi, 1e-6);

    // Check the contents of the first row.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.770844212e-1, pattern.g_theta[0].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-3.556198490e-1, pattern.g_theta[0].y, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.446807267, pattern.g_phi[0].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.374663044, pattern.g_phi[0].y, 1e-6);

    // Check the contents of row 15.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0192064, pattern.g_theta[14].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.90594407, pattern.g_theta[14].y, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.013445964, pattern.g_phi[14].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.151916707, pattern.g_phi[14].y, 1e-6);

    // Check the contents of the last row.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.05954787, pattern.g_theta[29].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.018560778, pattern.g_theta[29].y, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.208770322e-6, pattern.g_phi[29].x, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.593518683e-5, pattern.g_phi[29].y, 1e-6);

    // Remove the file.
    printf("Antenna data loaded successfully.\n");
    remove(filename);

    // Free the memory.
    free(pattern.g_phi);
    free(pattern.g_theta);
}
