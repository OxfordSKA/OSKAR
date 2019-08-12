/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#ifndef OSKAR_DEVICE_H_
#define OSKAR_DEVICE_H_

/**
 * @file oskar_device.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Device;
#ifndef OSKAR_DEVICE_TYPEDEF_
#define OSKAR_DEVICE_TYPEDEF_
typedef struct oskar_Device oskar_Device;
#endif

#define PTR_SZ sizeof(void*)
#define INT_SZ sizeof(int)
#define FLT_SZ sizeof(float)
#define DBL_SZ sizeof(double)

typedef struct
{
    size_t size;
    const void* ptr;
} oskar_Arg;

/**
 * @brief Checks if a CUDA device error occurred.
 *
 * @details
 * This function checks to see if a CUDA device error occurred.
 *
 * @param[out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_device_check_error_cuda(int* status);

/**
 * @brief Ensures the work-group size is compatible for the compute device.
 *
 * @details
 * Note that this is only really useful for OpenCL devices.
 *
 * @param[in] location       Enumerated location type.
 * @param[in] dim            Dimension index (0-2).
 * @param[in,out] local_size Local size to check; dimension reduced if required.
 */
OSKAR_EXPORT
void oskar_device_check_local_size(int location, unsigned int dim,
        size_t local_size[3]);

/**
 * @brief Returns the device context for the current OpenCL device.
 *
 * @details
 * Returns the device context for the current OpenCL device.
 * Note that this function will initialise the device if necessary.
 */
OSKAR_EXPORT
struct _cl_context* oskar_device_context_cl(void);

/**
 * @brief Returns a handle to the specified OpenCL device.
 *
 * @details
 * Returns a handle to the specified OpenCL device, or NULL if out of range.
 * Note that this function will initialise the device if necessary.
 * Note that this is intended to be read-only.
 *
 * @param[in] id          Device ID.
 */
OSKAR_EXPORT
const oskar_Device* oskar_device_cl(int id);

/**
 * @brief Creates a new, empty device info handle and returns it.
 *
 * @details
 * Creates a new, empty device info handle and returns it.
 */
OSKAR_EXPORT
oskar_Device* oskar_device_create(void);

/**
 * @brief Releases resources held in the specified device structure.
 *
 * @details
 * Releases resources held in the specified device structure.
 *
 * @param[in] device    Device handle.
 */
OSKAR_EXPORT
void oskar_device_free(oskar_Device* device);

/**
 * @brief Helper function to calculate the global grid size.
 *
 * @details
 * This helper function calculates the required global grid size, given the
 * problem size and local work group size.
 *
 * It is equivalent to finding the next largest multiple of the local size.
 */
OSKAR_EXPORT
size_t oskar_device_global_size(size_t num, size_t local_size);

/**
 * @brief Initialises OpenCL device contexts.
 *
 * @details
 * (Re-)creates contexts to work with OpenCL devices.
 *
 * The environment variables OSKAR_CL_DEVICE_TYPE and OSKAR_CL_DEVICE_VENDOR
 * will be read if set:
 *
 * OSKAR_CL_DEVICE_TYPE may be "GPU", "CPU", "Accelerator" or "All".
 * OSKAR_CL_DEVICE_VENDOR should be the name of the device vendor to use.
 */
OSKAR_EXPORT
void oskar_device_init_cl();

/**
 * @brief Returns true if current compute device is a CPU.
 *
 * @details
 * Returns true if current compute device is a CPU.
 * Note that this is only really useful for OpenCL devices.
 *
 * @param[in] location       Enumerated location type.
 */
OSKAR_EXPORT
int oskar_device_is_cpu(int location);

/**
 * @brief Returns true if current compute device is a GPU.
 *
 * @details
 * Returns true if current compute device is a GPU.
 * Note that this is only really useful for OpenCL devices.
 *
 * @param[in] location       Enumerated location type.
 */
OSKAR_EXPORT
int oskar_device_is_gpu(int location);

/**
 * @brief Returns true if current compute device is an NVIDIA GPU.
 *
 * @details
 * Returns true if current compute device is a NVIDIA GPU.
 * Note that this is only really useful for OpenCL devices.
 *
 * @param[in] location       Enumerated location type.
 */
OSKAR_EXPORT
int oskar_device_is_nv(int location);

/**
 * @brief Launches a compute kernel.
 *
 * @details
 * Launches a compute kernel on the current CUDA or OpenCL device.
 *
 * @param[in] name           Name of the kernel to launch.
 * @param[in] location       Either OSKAR_GPU or OSKAR_CL.
 * @param[in] num_dims       Number of kernel work group dimensions (up to 3).
 * @param[in] local_size[3]  3D work group, or thread block size.
 * @param[in] global_size[3] Total size of 3D grid. (Multiple of local size.)
 * @param[in] num_args       Number of kernel arguments, excluding local memory.
 * @param[in] args           Array of oskar_Arg kernel arguments.
 * @param[in] num_local_args Number of local memory arguments.
 * @param[in] arg_size_local Size of each local memory argument, in bytes.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_device_launch_kernel(const char* name, int location,
        int num_dims, size_t local_size[3], size_t global_size[3],
        size_t num_args, const oskar_Arg* arg,
        size_t num_local_args, const size_t* arg_size_local, int* status);

/**
 * @brief Returns the name of the specified device.
 *
 * @details
 * This function returns the name of the specified device.
 * Call free() with the returned name when it is no longer required.
 *
 * @param[in] location  Enumerated device location.
 * @param[in] id        Device ID.
 */
OSKAR_EXPORT
char* oskar_device_name(int location, int id);

/**
 * @brief Returns the default command queue for the current OpenCL device.
 *
 * @details
 * Returns the default command queue for the current OpenCL device.
 * Note that this function will initialise the device if necessary.
 */
OSKAR_EXPORT
struct _cl_command_queue* oskar_device_queue_cl(void);

/**
 * @brief
 * Returns the flag specifying whether or not double precision is required.
 *
 * @details
 * Returns the flag specifying whether or not double precision is required.
 */
OSKAR_EXPORT
int oskar_device_require_double(void);

/**
 * @brief Resets all devices.
 *
 * @details
 * Resets all devices and releases all resources held by them.
 * Use only at the end of an application.
 */
OSKAR_EXPORT
void oskar_device_reset_all(void);

/**
 * @brief Sets the device to use for subsequent device calls.
 *
 * @details
 * Sets the device to use for subsequent device calls.
 *
 * @param[in] location    Enumerated device location.
 * @param[in] id          Device ID.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_device_set(int location, int id, int* status);

/**
 * @brief Sets whether devices supporting double precision are required.
 *
 * @details
 * Sets a flag to determine whether devices that support double precision
 * are required or not.
 *
 * The default assumption is that only devices supporting double precision
 * can be used by OSKAR.
 *
 * Call this function BEFORE querying or initialising any devices.
 *
 * @param[in] flag If set, double precision is required;
 *                 if clear, single precision is sufficient.
 */
OSKAR_EXPORT
void oskar_device_set_require_double_precision(int flag);

/**
 * @brief
 * Returns the flag specifying whether or not 64-bit atomics are supported.
 *
 * @details
 * Returns the flag specifying whether or not 64-bit atomics are supported
 * on the current device.
 *
 * @param[in] location       Enumerated location type.
 */
OSKAR_EXPORT
int oskar_device_supports_atomic64(int location);

/**
 * @brief
 * Returns the flag specifying whether or not double precision is supported.
 *
 * @details
 * Returns the flag specifying whether or not double precision is supported
 * on the current device.
 *
 * @param[in] location       Enumerated location type.
 */
OSKAR_EXPORT
int oskar_device_supports_double(int location);

#ifdef __cplusplus
}
#endif

#include <utility/oskar_device_count.h>
#include <utility/oskar_device_create_list.h>
#include <utility/oskar_device_get_info.h>
#include <utility/oskar_device_log.h>

#endif /* include guard */
