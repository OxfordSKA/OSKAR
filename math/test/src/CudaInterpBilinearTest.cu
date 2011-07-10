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

#include "math/test/CudaInterpBilinearTest.h"
#include "math/oskar_math_cuda_interp_bilinear.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaInterpBilinearTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaInterpBilinearTest::tearDown()
{
}

/**
 * @details
 * Tests bilinear interpolation using CUDA.
 */
void CudaInterpBilinearTest::test_method()
{
	int width = 3;
	int height = 3;
	float data[] = {
			0.1, 0.8, 1.0,
			0.5, 2.0, 0.6,
			0.2, 1.1, 0.7
	};

	// Set up positions.
	int factor = 40;
	int nw = width * factor;
	int nh = height * factor;
	int n = nw * nh;
	float2* pos = (float2*)malloc(n * sizeof(float2));
	for (int i = 0, h = 0; h < nh; h++)
	{
		for (int w = 0; w < nw; w++)
		{
			pos[i].x = 0.5 + (width - 1) * float(w) / float(nw-1);
			pos[i].y = 0.5 + (height - 1) * float(h) / float(nh-1);
//			printf("%5d %10.4f %10.4f\n", i, pos[i].x, pos[i].y);
			i++;
		}
	}

	// Copy data to device.
	float* data_d;
	size_t pitch;
	cudaMallocPitch((void**)&data_d, &pitch, width, height);
	cudaMemcpy2D(data_d, pitch, data, width * sizeof(float),
			width * sizeof(float), height, cudaMemcpyHostToDevice);

	// Copy positions to device.
	float2* pos_d;
	cudaMalloc((void**)&pos_d, n * sizeof(float2));
	cudaMemcpy(pos_d, pos, n * sizeof(float2), cudaMemcpyHostToDevice);

	// Allocate result.
	float* output = (float*)malloc(n * sizeof(float));
	float* output_d;
	cudaMalloc((void**)&output_d, n * sizeof(float));

	// Interpolate.
	printf("Starting interpolation...\n");
	int err;
	TIMER_START
	err = oskar_math_cuda_interp_bilinear_float(width, height, pitch,
			data_d, n, pos_d, output_d);
	TIMER_STOP("Finished interpolation (%d points)", n)
	if (err != 0)
		printf("CUDA error, code %d\n", err);
	else
		printf("Interpolation successful!\n");

	// Copy result back.
	cudaMemcpy(output, output_d, n * sizeof(float), cudaMemcpyDeviceToHost);

	// Print result to file.
	FILE* file = fopen("bilinear_interp_test.dat", "w");
	for (int i = 0, h = 0; h < height * factor; h++)
	{
		for (int w = 0; w < width * factor; w++)
		{
			fprintf(file, "%6.3f", output[i]);
			i++;
		}
		fprintf(file, "\n");
	}
	fclose(file);

	// Free memory.
	free(output);
	free(pos);
	cudaFree(output_d);
	cudaFree(pos_d);
	cudaFree(data_d);
}

