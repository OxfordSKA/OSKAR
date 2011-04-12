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

#include "cuda/test/CudaVectorAddTest.h"
#include "cuda/oskar_cuda_vecadd.h"
#include <cmath>
#include <cstdlib>

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(CudaVectorAddTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaVectorAddTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaVectorAddTest::tearDown()
{
}

/**
 * @details
 * Tests vector addition using CUDA.
 */
void CudaVectorAddTest::test_method()
{
    printf("Vector addition test... ");

    // Define number of elements in vectors.
    int n = 50000;

    // Allocate vectors.
    std::vector<float> a(n), b(n), c(n, 0.0f);

    // Fill input vectors with random numbers.
    for (int i = 0; i < n; ++i) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Add the two vectors.
    oskar_cuda_vecadd(n, &a[0], &b[0], &c[0]);

    // Verify result.
    int i;
    for (i = 0; i < n; ++i) {
        float sum = a[i] + b[i];
        if (fabs(c[i] - sum) > 1e-5)
            break;
    }

    // Pass or fail.
    if (i != n) {
        printf("FAILED.\n");
        printf("i= %d, a[i]= %f, b[i]= %f, c[i]= %f\n", i, a[i], b[i], c[i]);
        CPPUNIT_FAIL("Vector addition test failed.");
    } else {
        printf("PASSED.\n");
    }
}
