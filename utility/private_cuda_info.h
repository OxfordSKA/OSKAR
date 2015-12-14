/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_CUDA_INFO_H_
#define OSKAR_PRIVATE_CUDA_INFO_H_

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_CudaDeviceInfo
{
    char name[256];            /* String holding device name. */
    union {
        struct {
            int major;         /* Compute capability, major version. */
            int minor;         /* Compute capability, minor version. */
        } capability;
        int version[2];
    } compute;
    int supports_double;       /* True if device supports double precision. */
    int global_memory_size;    /* Total size in kiB. */
    int free_memory;           /* Free memory in kiB. */
    int num_multiprocessors;   /* Number of multiprocessors. */
    int num_cores;             /* Number of CUDA cores. */
    int gpu_clock;             /* GPU clock speed in kHz. */
    int memory_clock;          /* Memory clock speed in kHz. */
    int memory_bus_width;      /* Memory bus width in bits. */
    int level_2_cache_size;    /* Cache size in bytes. */
    int shared_memory_size;    /* Shared memory per block in bytes. */
    int num_registers;         /* Number of registers per block. */
    int warp_size;             /* Warp size. */
    int max_threads_per_block; /* Maximum number of threads per block. */
    int max_threads_dim[3];    /* Maximum threads per dimension. */
    int max_grid_size[3];      /* Maximum grid size per dimension. */
};
typedef struct oskar_CudaDeviceInfo oskar_CudaDeviceInfo;

struct oskar_CudaInfo
{
    int num_devices;              /* Number of installed CUDA devices. */
    int driver_version;           /* CUDA driver version. */
    int runtime_version;          /* CUDA runtime version. */
    oskar_CudaDeviceInfo* device; /* Array of device info structures. */
};
#ifndef OSKAR_CUDA_INFO_TYPEDEF_
#define OSKAR_CUDA_INFO_TYPEDEF_
typedef struct oskar_CudaInfo oskar_CudaInfo;
#endif /* OSKAR_CUDA_INFO_TYPEDEF_ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_CUDA_INFO_H_ */
