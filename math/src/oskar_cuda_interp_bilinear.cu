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

#include "math/oskar_cuda_interp_bilinear.h"
#include "oskar_global.h"

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
template <typename InputType, typename CoordType, typename OutputType>
__global__ void oskar_cudak_interp_bilinear(const CoordType size_x,
        const CoordType size_y, const int n, const CoordType* pos_x,
        const CoordType* pos_y, OutputType* out)
{
    const int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i < n)
    {
        // Get normalised coordinates from global memory.
    	float2 p = make_float2((float)pos_x[i], (float)pos_y[i]);

        // Re-scale coordinates to allow for 0.5-pixel offsets.
        p.x = 0.5f + (size_x - 1) * p.x;
        p.y = 0.5f + (size_y - 1) * p.y;

        // Perform interpolated texture lookup.
        out[i] = tex2D(texture_ref<InputType>(), p.x, p.y);
    }
}

// Kernel template specialisation for complex double output.
template <>
__global__ void oskar_cudak_interp_bilinear<float2, double, double2>(
        const double size_x, const double size_y, const int n,
        const double* pos_x, const double* pos_y, double2* out)
{
    const int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (i < n)
    {
        // Get normalised coordinates from global memory.
    	float2 p = make_float2((float)pos_x[i], (float)pos_y[i]);

        // Re-scale coordinates to allow for 0.5-pixel offsets.
        p.x = 0.5f + (size_x - 1) * p.x;
        p.y = 0.5f + (size_y - 1) * p.y;

        // Perform interpolated texture lookup.
        float2 t1 = tex2D(texture_ref<float2>(), p.x, p.y);

        // Promote result to double.
        double2 t2 = make_double2(t1.x, t1.y);
        out[i] = t2;
    }
}

// Kernel wrapper.
template <typename InputType, typename CoordType, typename OutputType>
int oskar_cuda_interp_bilinear(int size_x, int size_y, int pitch,
        const InputType* d_input, int n, const CoordType* d_pos_x,
        const CoordType* d_pos_y, OutputType* d_output)
{
    // Prepare the texture reference from the look-up table.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<InputType>();
    texture<InputType, 2>& ref = texture_ref<InputType>();
    ref.filterMode = cudaFilterModeLinear;
    ref.normalized = false;
    cudaError_t errCuda = cudaBindTexture2D(0, &ref, d_input, &channelDesc,
            size_x, size_y, pitch);
    if (errCuda != cudaSuccess) return errCuda;

    // Launch the kernel.
    const int thd = 512; // Using more than this won't work with CUDA 1.3.
    const int blk = (n + thd - 1) / thd;
    oskar_cudak_interp_bilinear<InputType, CoordType, OutputType>
    		OSKAR_CUDAK_CONF(blk, thd)
    		(size_x, size_y, n, d_pos_x, d_pos_y, d_output);
    cudaDeviceSynchronize();

    // Unbind texture.
    cudaUnbindTexture(&ref);

    // Return error code.
    return cudaPeekAtLastError();
}

// Single precision.
extern "C"
int oskar_cuda_interp_bilinear_f(int size_x, int size_y, int pitch,
        const float* d_input, int n, const float* d_pos_x,
        const float* d_pos_y, float* d_output)
{
    return oskar_cuda_interp_bilinear<float, float, float>(size_x, size_y,
            pitch, d_input, n, d_pos_x, d_pos_y, d_output);
}

// Single precision complex.
extern "C"
int oskar_cuda_interp_bilinear_c(int size_x, int size_y, int pitch,
        const float2* d_input, int n, const float* d_pos_x,
        const float* d_pos_y, float2* d_output)
{
    return oskar_cuda_interp_bilinear<float2, float, float2>(size_x, size_y,
            pitch, d_input, n, d_pos_x, d_pos_y, d_output);
}

// Double precision.
extern "C"
int oskar_cuda_interp_bilinear_d(int size_x, int size_y, int pitch,
        const float* d_input, int n, const double* d_pos_x,
        const double* d_pos_y, double* d_output)
{
    return oskar_cuda_interp_bilinear<float, double, double>(size_x, size_y,
            pitch, d_input, n, d_pos_x, d_pos_y, d_output);
}

// Double precision complex.
extern "C"
int oskar_cuda_interp_bilinear_z(int size_x, int size_y, int pitch,
        const float2* d_input, int n, const double* d_pos_x,
        const double* d_pos_y, double2* d_output)
{
    return oskar_cuda_interp_bilinear<float2, double, double2>(size_x, size_y,
            pitch, d_input, n, d_pos_x, d_pos_y, d_output);
}
