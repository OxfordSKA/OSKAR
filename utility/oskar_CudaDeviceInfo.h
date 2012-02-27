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

#ifndef OSKAR_CUDA_DEVICE_INFO_H_
#define OSKAR_CUDA_DEVICE_INFO_H_

/**
 * @file oskar_cuda_device_info.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure to hold CUDA device information.
 *
 * @details
 * This structure holds information about a CUDA device.
 */
struct oskar_CudaDeviceInfo
{
    char name[256];            /**< String holding device name. */
    union {
        struct {
            int major;         /**< Compute capability, major version. */
            int minor;         /**< Compute capability, minor version. */
        } capability;
        int version[2];
    } compute;
    int supports_double;       /**< True if device supports double precision. */
    int global_memory_size;    /**< Total size in kiB. */
    int num_multiprocessors;   /**< Number of multiprocessors. */
    int num_cores;             /**< Number of CUDA cores. */
    int gpu_clock;             /**< GPU clock speed in kHz. */
    int memory_clock;          /**< Memory clock speed in kHz. */
    int memory_bus_width;      /**< Memory bus width in bits. */
    int level_2_cache_size;    /**< Cache size in bytes. */
    int shared_memory_size;    /**< Shared memory per block in bytes. */
    int num_registers;         /**< Number of registers per block. */
    int warp_size;             /**< Warp size. */
    int max_threads_per_block; /**< Maximum number of threads per block. */
    int max_threads_dim[3];    /**< Maximum threads per dimension. */
    int max_grid_size[3];      /**< Maximum grid size per dimension. */
};
typedef struct oskar_CudaDeviceInfo oskar_CudaDeviceInfo;

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CUDA_DEVICE_INFO_H_ */
