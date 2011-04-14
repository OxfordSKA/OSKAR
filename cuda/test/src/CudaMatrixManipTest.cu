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

#include "cuda/test/CudaMatrixManipTest.h"
#include "cuda/kernels/oskar_cudak_cmatadd.h"
#include "cuda/kernels/oskar_cudak_cmatmul.h"
#include "cuda/kernels/oskar_cudak_cmatset.h"
#include <cmath>
#include <cstdlib>

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(CudaMatrixManipTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaMatrixManipTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaMatrixManipTest::tearDown()
{
}

/**
 * @details
 * Tests vector addition using CUDA.
 */
void CudaMatrixManipTest::test_method()
{
    printf("Matrix manipulation test... ");

    int na = 25;

    // Allocate device memory.
    float2 *visd;
    cudaMalloc((void**)&visd, na * na * sizeof(float2));

    // Allocate host memory.
    float* vis = (float*)malloc(2 * na * na * sizeof(float));

    // Define thread block.
    dim3 vThd(16, 16); // Antennas, antennas.
    dim3 vBlk((na + vThd.x - 1) / vThd.x, (na + vThd.y - 1) / vThd.y);

    // Clear matrix.
    oskar_cudak_cmatset <<<vBlk, vThd>>> (
            na, na, make_float2(2.0, 2.0), visd);
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        CPPUNIT_FAIL("Test failed.");
    }

    // Copy result to host.
    cudaMemcpy(vis, visd, na * na * sizeof(float2), cudaMemcpyDeviceToHost);

    // Print out matrix contents.
    for (int i = 0; i < 2 * na * na; ++i) {
        if (vis[i] != 2.0f) {
            printf("Matrix corrupted.\n");
        }
    }

    // Free memory.
    free(vis);
    cudaFree(visd);
}
