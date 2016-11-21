/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#ifndef OSKAR_DEVICE_UTILS_H_
#define OSKAR_DEVICE_UTILS_H_

/**
 * @file oskar_device_utils.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Checks if a device error occurred.
 *
 * @details
 * This function checks to see if a device error occurred.
 * In debug builds, a call to cudaDeviceSynchronize() is made before the check.
 *
 * @param[out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_device_check_error(int* status);

/**
 * @brief
 * Returns the number of devices on the system.
 *
 * @details
 * This simply calls cudaDeviceReset() if CUDA is available.
 */
OSKAR_EXPORT
int oskar_device_count(int* status);

/**
 * @brief
 * Returns the amount of free memory on the device.
 *
 * @details
 * This simply calls cudaMemGetInfo() if CUDA is available.
 */
OSKAR_EXPORT
void oskar_device_mem_info(size_t* mem_free, size_t* mem_total);

/**
 * @brief
 * Returns the name of the specified device.
 *
 * @details
 * The name of the device specified by \p device_id is returned as a string.
 * Use free() to deallocate the string.
 */
OSKAR_EXPORT
char* oskar_device_name(int device_id);

/**
 * @brief
 * Resets the current device.
 *
 * @details
 * This simply calls cudaDeviceReset() if CUDA is available.
 */
OSKAR_EXPORT
void oskar_device_reset(void);

/**
 * @brief
 * Sets the device to use for subsequent device calls.
 *
 * @details
 * Sets the device to use for subsequent device calls.
 * This simply calls cudaSetDevice() if CUDA is available.
 *
 * @param[in] id          CUDA device ID.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_device_set(int id, int* status);

/**
 * @brief
 * Synchronises the current device.
 *
 * @details
 * This simply calls cudaDeviceSynchronize() if CUDA is available.
 */
OSKAR_EXPORT
void oskar_device_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DEVICE_UTILS_H_ */
