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

#include "math/test/Matrix3Test.h"
#include "math/Matrix3.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(Matrix3Test);

/**
 * @details
 * Sets up the context before running each test method.
 */
void Matrix3Test::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void Matrix3Test::tearDown()
{
}

/**
 * @details
 * Tests identity.
 */
void Matrix3Test::test_identity()
{
    float matrix[9];
    Matrix3::identity(matrix);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matrix[0], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matrix[4], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matrix[8], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[1], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[2], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[3], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[5], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[6], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matrix[7], 1e-4);
}

/**
 * @details
 * Tests multiplication.
 */
void Matrix3Test::test_multiply()
{
    float identity[9];
    Matrix3::identity(identity);

    float rotateX[9];
    float angle = 30 * DEG2RAD;
    Matrix3::rotationX(rotateX, angle);

    // Test multiply with identity matrix.
    {
        float result[9];
        Matrix3::multiplyMatrix3(result, identity, rotateX);

        for (int i = 0; i < 9; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(result[i], rotateX[i], 1e-4);
        }
    }

    // Test two rotations.
    {
        float rotateX2[9], result[9];
        Matrix3::rotationX(rotateX2, 2 * angle);
        Matrix3::multiplyMatrix3(result, rotateX, rotateX);

        for (int i = 0; i < 9; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(result[i], rotateX2[i], 1e-4);
        }
    }
}

/**
 * @details
 * Tests rotation.
 */
void Matrix3Test::test_rotation()
{
    // Test rotation around x-axis.
    {
        float matrix[9], matrixX[9];
        float axis[3] = {1, 0, 0};
        float angle = 30 * DEG2RAD;
        Matrix3::rotation(matrix, axis, angle);
        Matrix3::rotationX(matrixX, angle);

        for (int i = 0; i < 9; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(matrixX[i], matrix[i], 1e-4);
        }
    }

    // Test rotation around y-axis.
    {
        float matrix[9], matrixY[9];
        float axis[3] = {0, 1, 0};
        float angle = 30 * DEG2RAD;
        Matrix3::rotation(matrix, axis, angle);
        Matrix3::rotationY(matrixY, angle);

        for (int i = 0; i < 9; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(matrixY[i], matrix[i], 1e-4);
        }
    }

    // Test rotation around z-axis.
    {
        float matrix[9], matrixZ[9];
        float axis[3] = {0, 0, 1};
        float angle = 30 * DEG2RAD;
        Matrix3::rotation(matrix, axis, angle);
        Matrix3::rotationZ(matrixZ, angle);

        for (int i = 0; i < 9; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(matrixZ[i], matrix[i], 1e-4);
        }
    }
}

/**
 * @details
 * Tests scaling.
 */
void Matrix3Test::test_scale()
{
    float matrix[9];
    float s[3] = {1, 2, 3};
    Matrix3::scaling(matrix, s);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matrix[0], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, matrix[4], 1e-4);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, matrix[8], 1e-4);
}
