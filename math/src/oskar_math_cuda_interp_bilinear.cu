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

#include "math/oskar_math_cuda_interp_bilinear.h"
#include "utility/oskar_util_cuda_eclipse.h"

// Texture references must be static global variables,
// hence kernel and all device code is in this file.
static texture<float, 2> texture_ref_float;
static texture<float2, 2> texture_ref_float2;

// Template declaration: returns a reference to the texture.
template <typename T>
inline __device__ __host__ texture<T, 2>& texture_ref();

// Template specialisation: float.
template <>
inline __device__ __host__ texture<float, 2>& texture_ref<float>()
{
	return texture_ref_float;
}

// Template specialisation: float2.
template <>
inline __device__ __host__ texture<float2, 2>& texture_ref<float2>()
{
	return texture_ref_float2;
}

// Kernel.
template <typename T>
__global__ void oskar_bilinear_kernel(int n, const float2* pos, T* out)
{
	const int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i < n)
	{
		float2 p = pos[i];
		out[i] = tex2D(texture_ref<T>(), p.x, p.y);
	}
}

// Kernel wrapper.
template <typename T>
int oskar_math_cuda_interp_bilinear(int width, int height, int pitch,
		const T* input, int n, const float2* pos, T* output)
{
	// Prepare the texture reference from the look-up table.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	texture<T, 2>& ref = texture_ref<T>();
	ref.filterMode = cudaFilterModeLinear;
	ref.normalized = false;
	cudaError_t errCuda = cudaBindTexture2D(0, &ref, input, &channelDesc,
			width, height, pitch);
	if (errCuda != cudaSuccess) return errCuda;

	// Launch the kernel.
	const int thd = 768;
	const int blk = (n + thd - 1) / thd;
	oskar_bilinear_kernel<T> <<< blk, thd >>> (n, pos, output);
	cudaDeviceSynchronize();
	errCuda = cudaPeekAtLastError();
	if (errCuda != cudaSuccess) return errCuda;

	// Unbind texture.
	cudaUnbindTexture(&ref);

	// Return 0 on success.
	return 0;
}

int oskar_math_cuda_interp_bilinear_float(int width, int height, int pitch,
		const float* input, int n, const float2* pos, float* output)
{
	return oskar_math_cuda_interp_bilinear<float>(width, height, pitch, input,
			n, pos, output);
}

int oskar_math_cuda_interp_bilinear_complex(int width, int height, int pitch,
		const float2* input, int n, const float2* pos, float2* output)
{
	return oskar_math_cuda_interp_bilinear<float2>(width, height, pitch, input,
				n, pos, output);
}
