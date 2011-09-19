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

#ifndef OSKAR_UTIL_CUDA_MEMORY_H_
#define OSKAR_UTIL_CUDA_MEMORY_H_

/**
 * @file oskar_util_cuda_memory.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Allocates memory on the graphics card.
 *
 * @details
 * This function allocates a block of memory on the graphics card and returns
 * a pointer to it.
 *
 * @param[in,out] ptr Pointer to memory block.
 * @param[in] size Size (in bytes) of block to allocate.
 */
OSKAR_EXPORT
void oskar_cuda_malloc(void** ptr, unsigned size);

/**
 * @brief
 * Allocates an array of doubles on the graphics card.
 *
 * @details
 * This function allocates an array of doubles on the graphics card and returns
 * a pointer to it.
 *
 * @param[in,out] ptr Pointer to memory block.
 * @param[in] n Number of items to allocate.
 */
OSKAR_EXPORT
void oskar_cuda_malloc_double(double** ptr, unsigned n);

/**
 * @brief
 * Allocates an array of floats on the graphics card.
 *
 * @details
 * This function allocates an array of floats on the graphics card and returns
 * a pointer to it.
 *
 * @param[in,out] ptr Pointer to memory block.
 * @param[in] n Number of items to allocate.
 */
OSKAR_EXPORT
void oskar_cuda_malloc_float(float** ptr, unsigned n);

/**
 * @brief
 * Allocates an array of integers on the graphics card.
 *
 * @details
 * This function allocates an array of integers on the graphics card and
 * returns a pointer to it.
 *
 * @param[in,out] ptr Pointer to memory block.
 * @param[in] n Number of items to allocate.
 */
OSKAR_EXPORT
void oskar_cuda_malloc_int(int** ptr, unsigned n);

/**
 * @brief
 * Copies a block of host memory to the device.
 *
 * @details
 * This function copies a block of memory from the host to the device.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] size Size (in bytes) of memory to copy.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_h2d(void* dest, const void* src, unsigned size);

/**
 * @brief
 * Copies an array of doubles from the host to the device.
 *
 * @details
 * This function copies an array of doubles from the host to the device.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_h2d_double(double* dest, const double* src, unsigned n);

/**
 * @brief
 * Copies an array of floats from the host to the device.
 *
 * @details
 * This function copies an array of floats from the host to the device.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_h2d_float(float* dest, const float* src, unsigned n);

/**
 * @brief
 * Copies an array of integers from the host to the device.
 *
 * @details
 * This function copies an array of integers from the host to the device.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_h2d_int(int* dest, const int* src, unsigned n);

/**
 * @brief
 * Copies a block of device memory to the host.
 *
 * @details
 * This function copies a block of memory from the device to the host.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] size Size (in bytes) of memory to copy.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_d2h(void* dest, const void* src, unsigned size);

/**
 * @brief
 * Copies an array of doubles from the device to the host.
 *
 * @details
 * This function copies an array of doubles from the device to the host.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_d2h_double(double* dest, const double* src, unsigned n);

/**
 * @brief
 * Copies an array of floats from the device to the host.
 *
 * @details
 * This function copies an array of floats from the device to the host.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_d2h_float(float* dest, const float* src, unsigned n);

/**
 * @brief
 * Copies an array of integers from the device to the host.
 *
 * @details
 * This function copies an array of integers from the device to the host.
 *
 * @param[in,out] dest Pointer to destination memory block.
 * @param[in] src Pointer to source memory block.
 * @param[in] n Number of items in the array.
 */
OSKAR_EXPORT
void oskar_cuda_memcpy_d2h_int(int* dest, const int* src, unsigned n);

/**
 * @brief
 * Frees memory on the graphics card.
 *
 * @details
 * This function frees a block of memory on the graphics card.
 *
 * @param[in] n Number of elements.
 */
OSKAR_EXPORT
void oskar_cuda_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_UTIL_CUDA_MEMORY_H_
