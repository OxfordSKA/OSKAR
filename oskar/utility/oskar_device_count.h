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

#ifndef OSKAR_DEVICE_COUNT_H_
#define OSKAR_DEVICE_COUNT_H_

/**
 * @file oskar_device_count.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the number of devices on the system.
 *
 * @details
 * This returns either the number of devices given by cudaGetDeviceCount(),
 * or the number of OpenCL devices.
 *
 * The platform to check can be manually specified either using the \p platform
 * parameter or the environment variable OSKAR_PLATFORM, which can take the
 * string values "CUDA" or "OpenCL". If specified, the \p platform parameter
 * will override the environment variable.
 *
 * If the device platform is not specified, the function returns the number
 * of CUDA devices if this is greater than 0, or the number of OpenCL devices
 * otherwise. The \p device_location parameter will be set to OSKAR_GPU (1) or
 * OSKAR_CL (2), respectively. If no CUDA or OpenCL devices are found, the
 * function returns 0 and the \p device_location parameter will be set to the
 * value of OSKAR_CPU (0). Note that the number of CPU cores are NOT returned
 * in this case! This is for discrete compute devices only.
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
 * This function only queries device information:
 * it does not initialise any devices.
 *
 * @param[in]  platform  If not NULL, either "CUDA" or "OpenCL".
 * @param[out] location  If not NULL, the enumerated device location.
 */
OSKAR_EXPORT
int oskar_device_count(const char* platform, int* location);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
