/*
 * Copyright (c) 2018-2019, The University of Oxford
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

#ifndef OSKAR_PRIVATE_DEVICE_H_
#define OSKAR_PRIVATE_DEVICE_H_

#include <oskar_global.h>
#include <stddef.h>

#ifdef OSKAR_HAVE_OPENCL
/* Needed for clCreateCommandQueue() */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

struct oskar_Device
{
    char *name, *vendor, *cl_version, *cl_driver_version;
    char platform_type, device_type;
    int index, init, is_nv;
    int supports_double, supports_atomic32, supports_atomic64;
    int max_compute_units, max_clock_freq_kHz;
    int memory_clock_freq_kHz, memory_bus_width;
    int cuda_driver_version, cuda_runtime_version, compute_capability[2];
    unsigned int num_cores, num_registers, warp_size;
    size_t global_mem_size, global_mem_free_size, global_mem_cache_size;
    size_t local_mem_size, max_mem_alloc_size;
    size_t max_local_size[3], max_work_group_size;
    struct oskar_DeviceKernels* kern;
    struct _cl_platform_id* platform_id;
    struct _cl_device_id* device_id_cl;
    struct _cl_context* context;
    struct _cl_command_queue* default_queue;
    struct _cl_program* program;
};
#ifndef OSKAR_DEVICE_TYPEDEF_
#define OSKAR_DEVICE_TYPEDEF_
typedef struct oskar_Device oskar_Device;
#endif

#endif /* include guard */
