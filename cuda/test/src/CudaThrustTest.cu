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

#include "cuda/test/CudaThrustTest.h"
#include <cmath>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define TIMER_ENABLE 1
#include "utility/timer.h"


// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(CudaThrustTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaThrustTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaThrustTest::tearDown()
{
}

struct is_positive {
    __host__ __device__
    bool operator()(const float x)
    {
        return x > 0.0;
    }
};

/**
 * @details
 * Tests CUDA Thrust performance.
 */
void CudaThrustTest::test_method()
{
    // Define number of elements in vectors.
    int n = 5000000;

    // Allocate vectors.
    std::vector<float> a(n), b(n), c(n, 0.0f);

    // Fill input vectors with random numbers.
    for (int i = 0; i < n; ++i) {
        a[i] = rand() / (float)RAND_MAX - 0.5;
        b[i] = rand() / (float)RAND_MAX - 0.5;
    }

    // Copy input data to device.
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));
    cudaMalloc((void**)&d_c, n * sizeof(float));
    cudaMemcpy(d_a, &a[0], n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b[0], n * sizeof(float), cudaMemcpyHostToDevice);

    int repeats = 50;

    thrust::device_ptr<float> out;
    TIMER_START
    for (int i = 0; i < repeats; ++i) {
        out = thrust::copy_if(thrust::device_pointer_cast(d_a),
                thrust::device_pointer_cast(d_a + n),
                thrust::device_pointer_cast(d_b),
                thrust::device_pointer_cast(d_c), is_positive());
    }
    TIMER_STOP("Finished Thrust GPU copy_if (%d iterations of %d elements)",
            repeats, n)
    ptrdiff_t diff = out - thrust::device_pointer_cast(d_c);
    std::cout << "Number of elements copied: " << diff << std::endl;

    cudaMemcpy(&c[0], d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

//    std::cout << "A (data): \n";
//    for (int i = 0; i < n; ++i)
//        std::cout << a[i] << std::endl;
//    std::cout << "\nB (stencil): \n";
//    for (int i = 0; i < n; ++i)
//        std::cout << b[i] << std::endl;
//    std::cout << "\nC (output): \n";
//    for (int i = 0; i < n; ++i)
//        std::cout << c[i] << std::endl;

    TIMER_START
    for (int i = 0; i < repeats; ++i) {
        thrust::copy_if(a.begin(), a.end(), b.begin(), c.begin(),
                is_positive());
    }
    TIMER_STOP("Finished Thrust CPU copy_if (%d iterations of %d elements)",
            repeats, n)

    float* pa = &a[0];
    float* pb = &b[0];
    float* pc = &c[0];
    TIMER_START
    for (int i = 0; i < repeats; ++i) {
        int k = 0; // current output index.
        for (int j = 0; j < n; ++j) {
            if (pb[j] > 0) {
                pc[k] = pa[j];
                k++;
            }
        }
    }
    TIMER_STOP("Finished CPU copy_if (%d iterations of %d elements)",
            repeats, n)

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
