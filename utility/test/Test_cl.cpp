/*
 * Copyright (c) 2016, The University of Oxford
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

#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

TEST(OpenCL, device_query)
{
    char* value;
    size_t value_len;
    cl_uint i, j, num_platforms, num_devices;
    cl_platform_id *platform_ids;
    cl_device_id* devices;
    cl_int error;

    // Get the OpenCL platform IDs.
    error = clGetPlatformIDs(0, 0, &num_platforms);
    if (error != CL_SUCCESS || num_platforms == 0)
    {
        printf("No OpenCL platform found!\n");
        return;
    }
    platform_ids = (cl_platform_id*) malloc(num_platforms *
        sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platform_ids, 0);

    for (i = 0; i < num_platforms; i++)
    {
        // Get all devices on the platform.
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        // Loop over devices.
        for (j = 0; j < num_devices; j++)
        {
            // Print device name.
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &value_len);
            value = (char*) malloc(value_len);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, value_len, value, NULL);
            printf("%d.%d. Device: %s\n", i+1, j+1, value);
            free(value);

            // Print device vendor.
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, NULL, &value_len);
            value = (char*) malloc(value_len);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, value_len, value, NULL);
            printf("    Device vendor: %s\n", value);
            free(value);

            // Print device type.
            cl_device_type type;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            if (type & CL_DEVICE_TYPE_CPU)
                printf("    Device type: %s\n", "CL_DEVICE_TYPE_CPU");
            if (type & CL_DEVICE_TYPE_GPU)
                printf("    Device type: %s\n", "CL_DEVICE_TYPE_GPU");
            if (type & CL_DEVICE_TYPE_ACCELERATOR)
                printf("    Device type: %s\n", "CL_DEVICE_TYPE_ACCELERATOR");
            if (type & CL_DEVICE_TYPE_DEFAULT)
                printf("    Device type: %s\n", "CL_DEVICE_TYPE_DEFAULT");
        }
        free(devices);
    }
    free(platform_ids);
}
