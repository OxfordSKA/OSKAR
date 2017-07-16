/*
 * Copyright (c) 2017, The University of Oxford
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

#include <oskar_global.h>

#ifdef OSKAR_HAVE_OPENCL
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OSKAR_HAVE_OPENCL

OSKAR_EXPORT
cl_command_queue oskar_cl_command_queue(void);

OSKAR_EXPORT
cl_context oskar_cl_context(void);

OSKAR_EXPORT
cl_device_id oskar_cl_device_id(void);

OSKAR_EXPORT
cl_kernel oskar_cl_kernel(const char*);

#endif

OSKAR_EXPORT
const char* oskar_cl_device_cl_version(void);

OSKAR_EXPORT
const char* oskar_cl_device_driver_version(void);

OSKAR_EXPORT
const char* oskar_cl_device_name(void);

OSKAR_EXPORT
void oskar_cl_free(void);

OSKAR_EXPORT
unsigned int oskar_cl_get_device(void);

/**
 * @brief
 * Initialises OpenCL contexts and command queues.
 *
 * @details
 * Creates contexts and command queues to work with OpenCL devices
 * matching the supplied parameters.
 *
 * The device type is optional. If not specified, all devices are selected.
 *
 * The device vendor name is optional.
 * More than one vendor name may be given in the same string using "|" or ","
 * as a delimiter.
 *
 * If either parameter is NULL, the values are checked from the
 * environment variables OSKAR_CL_DEVICE_TYPE and OSKAR_CL_DEVICE_VENDOR.
 *
 * @param[in] device_type    String containing required device type
 *                           (GPU, CPU or Accelerator). May be NULL.
 * @param[in] device_vendor  String containing required device vendor name.
 *                           May be NULL.
 */
OSKAR_EXPORT
void oskar_cl_init(const char* device_type, const char* device_vendor);

OSKAR_EXPORT
unsigned int oskar_cl_num_devices(void);

OSKAR_EXPORT
void oskar_cl_set_device(unsigned int device, int* status);

#ifdef __cplusplus
}
#endif

