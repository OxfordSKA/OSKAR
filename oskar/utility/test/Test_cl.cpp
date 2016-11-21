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
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, 0, &num_devices);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, num_devices, devices, 0);

        // Loop over devices.
        for (j = 0; j < num_devices; j++)
        {
            // Print device name.
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, 0, &value_len);
            value = (char*) malloc(value_len);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, value_len, value, 0);
            printf("%d.%d. Device: %s\n", i+1, j+1, value);
            free(value);

            // Print device vendor.
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, 0, &value_len);
            value = (char*) malloc(value_len);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, value_len, value, 0);
            printf("    Device vendor: %s\n", value);
            free(value);

            // Print device type.
            cl_device_type type;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, 0);
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

struct oskar_Context
{
    char* device_name;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
};
typedef struct oskar_Context oskar_Context;

void oskar_context_free(oskar_Context* h);

oskar_Context* oskar_context_create(const char* device_vendor,
        const char* device_type, int device_index)
{
    oskar_Context* h = 0;
    cl_uint i, j, num_platforms = 0, num_devices = 0;
    cl_platform_id *platforms;
    cl_device_id* devices;
    cl_device_type dev_type;
    cl_int current_device_index = 0, error = 0, found = 0;

    /* Set the device type to match. */
    if (!strncmp(device_type, "G", 1) || !strncmp(device_type, "g", 1))
        dev_type = CL_DEVICE_TYPE_GPU;
    else if (!strncmp(device_type, "C", 1) || !strncmp(device_type, "c", 1))
        dev_type = CL_DEVICE_TYPE_CPU;
    else if (!strncmp(device_type, "A", 1) || !strncmp(device_type, "a", 1))
        dev_type = CL_DEVICE_TYPE_ACCELERATOR;
    else
    {
        fprintf(stderr, "Unrecognised device type.\n");
        return 0;
    }

    /* Get the OpenCL platform IDs. */
    error = clGetPlatformIDs(0, 0, &num_platforms);
    if (error != CL_SUCCESS || num_platforms == 0)
    {
        fprintf(stderr, "No OpenCL platform found.\n");
        return 0;
    }
    h = (oskar_Context*) calloc(1, sizeof(oskar_Context));
    platforms = (cl_platform_id*) malloc(num_platforms *
        sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, 0);

    /* Loop over platforms. */
    for (i = 0; i < num_platforms; i++)
    {
        if (found || error) break;

        /* Get all device IDs on the platform that match the device type. */
        clGetDeviceIDs(platforms[i], dev_type, 0, 0, &num_devices);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[i], dev_type, num_devices, devices, 0);

        /* Loop over devices. */
        for (j = 0; j < num_devices; j++)
        {
            char* vendor;
            size_t len;
            if (found || error) break;

            /* Get device vendor name, and check for match. */
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, 0, &len);
            vendor = (char*) calloc(1, len);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, len, vendor, 0);
            if (!device_vendor || strlen(device_vendor) == 0 ||
                    !strcmp(vendor, device_vendor))
            {
                if (current_device_index == device_index)
                {
                    /* We found the device. */
                    found = 1;
                    h->device_id = devices[j];
                    clGetDeviceInfo(h->device_id, CL_DEVICE_NAME, 0, 0, &len);
                    h->device_name = (char*) calloc(1, len);
                    clGetDeviceInfo(
                            h->device_id, CL_DEVICE_NAME, len,
                            h->device_name, 0);

                    /* Create a context. */
                    if (!error)
                        h->context = clCreateContext(
                                0, 1, &h->device_id, 0, 0, &error);

                    /* Create a command queue. */
                    if (!error)
                        h->queue = clCreateCommandQueue(
                                h->context, h->device_id, 0, &error);
                }
                else
                {
                    /* Right vendor, but wrong device index: Keep looking. */
                    current_device_index++;
                }
            }
            free(vendor);
        }
        free(devices);
    }
    free(platforms);

    /* Return handle if we found the device and there was no error. */
    if (found && !error)
        return h;

    /* Otherwise, return nothing. */
    oskar_context_free(h);
    return 0;
}


void oskar_context_free(oskar_Context* h)
{
    if (!h) return;
    free(h->device_name);
    if (h->queue)
        clReleaseCommandQueue(h->queue);
    if (h->context)
        clReleaseContext(h->context);
    free(h);
}

