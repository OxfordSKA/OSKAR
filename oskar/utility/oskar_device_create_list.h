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

#ifndef OSKAR_DEVICE_CREATE_LIST_H_
#define OSKAR_DEVICE_CREATE_LIST_H_

/**
 * @file oskar_device_create_list.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns a list containing device information.
 *
 * @details
 * Returns a list containing device information about CUDA or OpenCL devices.
 * The device type is specified using the enumerated \p location parameter.
 *
 * For OpenCL, the environment variables OSKAR_CL_DEVICE_VENDOR
 * and OSKAR_CL_DEVICE_TYPE will be read if set:
 *
 * OSKAR_CL_DEVICE_TYPE may be "GPU", "CPU", "Accelerator" or "All".
 * OSKAR_CL_DEVICE_VENDOR should be the name of the device vendor to use.
 *
 * If the OpenCL environment variable(s) are not set, the function returns
 * information about OpenCL GPU devices if the number of GPUs is greater than 0.
 * If no GPUs are found, the function will return information about
 * OpenCL CPU devices instead.
 *
 * Note that this function only queries device information:
 * it does not initialise any devices.
 *
 * @param[in] location     Enumerated device location.
 * @param[out] num_devices Number of devices returned in the list.
 */
OSKAR_EXPORT
oskar_Device** oskar_device_create_list(int location, int* num_devices);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
