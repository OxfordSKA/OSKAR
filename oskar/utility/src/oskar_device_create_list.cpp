/*
 * Copyright (c) 2018-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "utility/private_device.h"
#include "utility/oskar_device.h"

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_thread.h"

static oskar_Device** oskar_device_create_list_cl(const char* device_type,
        const char* device_vendor, int* device_count);

struct oskar_LocalMutex
{
    oskar_Mutex* m;
    oskar_LocalMutex()  { this->m = oskar_mutex_create(); }
    ~oskar_LocalMutex() { oskar_mutex_free(this->m); }
    void lock() const   { oskar_mutex_lock(this->m); }
    void unlock() const { oskar_mutex_unlock(this->m); }
};
static oskar_LocalMutex mutex_; // NOLINT: This constructor will not throw.

oskar_Device** oskar_device_create_list(int location, int* num_devices)
{
    oskar_Device** devices = 0;
    *num_devices = 0;
    if (location == OSKAR_GPU)
    {
        *num_devices = oskar_device_count("CUDA", 0);
        if (*num_devices == 0) return 0;
        devices = (oskar_Device**) calloc(*num_devices, sizeof(oskar_Device*));
        for (int i = 0; i < *num_devices; ++i)
        {
            devices[i] = oskar_device_create();
            devices[i]->index = i;
            oskar_device_get_info_cuda(devices[i]);
        }
    }
    else if (location & OSKAR_CL)
    {
        static volatile int checked_env_ = 0;
        static std::string type_, vendor_;
        if (!checked_env_)
        {
            mutex_.lock();
            if (!checked_env_)
            {
                const char* type = getenv("OSKAR_CL_DEVICE_TYPE");
                if (type) type_ = std::string(type);
                const char* vendor = getenv("OSKAR_CL_DEVICE_VENDOR");
                if (vendor) vendor_ = std::string(vendor);
                checked_env_ = 1;
            }
            mutex_.unlock();
        }
        if (type_.size() > 0)
        {
            devices = oskar_device_create_list_cl(
                    type_.c_str(), vendor_.c_str(), num_devices);
        }
        if (*num_devices == 0)
        {
            devices = oskar_device_create_list_cl(
                    "GPU", vendor_.c_str(), num_devices);
            if (*num_devices == 0)
            {
                devices = oskar_device_create_list_cl(
                        "CPU", vendor_.c_str(), num_devices);
            }
        }
    }
    return devices;
}


static oskar_Device** oskar_device_create_list_cl(const char* device_type,
        const char* device_vendor, int* device_count)
{
    oskar_Device** device_list = 0;
    *device_count = 0;
#ifdef OSKAR_HAVE_OPENCL
    cl_uint num_platforms = 0, num_devices = 0;
    cl_device_type dev_type = CL_DEVICE_TYPE_ALL;
    cl_int error = 0;

    // Set the vendor(s) and device type to match.
    std::vector<std::string> vendors;
    if (device_vendor && strlen(device_vendor) > 0)
    {
        std::string vd = std::string(device_vendor);
        std::string delims = "|,";
        for (size_t k = 0; k < vd.size(); ++k) vd[k] = toupper(vd[k]);
        size_t start = vd.find_first_not_of(delims), end = 0;
        while ((end = vd.find_first_of(delims, start)) != std::string::npos)
        {
            vendors.push_back(vd.substr(start, end - start));
            start = vd.find_first_not_of(delims, end);
        }
        if (start != std::string::npos)
        {
            vendors.push_back(vd.substr(start));
        }
    }
    if (device_type && strlen(device_type) > 0)
    {
        std::string tp = std::string(device_type);
        for (size_t k = 0; k < tp.size(); ++k) tp[k] = toupper(tp[k]);
        if (!strncmp(tp.c_str(), "G", 1))
        {
            dev_type = CL_DEVICE_TYPE_GPU;
        }
        else if (!strncmp(tp.c_str(), "C", 1))
        {
            dev_type = CL_DEVICE_TYPE_CPU;
        }
        else if (!strncmp(tp.c_str(), "AC", 2))
        {
            dev_type = CL_DEVICE_TYPE_ACCELERATOR;
        }
        else
        {
            dev_type = CL_DEVICE_TYPE_ALL;
        }
    }

    // The first OpenCL call doesn't seem to be thread-safe.
    mutex_.lock();
    error = clGetPlatformIDs(0, 0, &num_platforms);
    mutex_.unlock();
    if (num_platforms == 0) return device_list;
    if (error != CL_SUCCESS)
    {
        oskar_log_error(0, "clGetPlatformIDs error (%d).", error);
        return device_list;
    }

    // Get platform and device IDs.
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, &platforms[0], 0);
    for (int p = (int)num_platforms - 1; p >= 0; p--)
    {
        clGetDeviceIDs(platforms[p], dev_type, 0, 0, &num_devices);
        if (num_devices == 0) continue;
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[p], dev_type, num_devices, &devices[0], 0);
        for (int d = (int)num_devices - 1; d >= 0; d--)
        {
            oskar_Device* device = oskar_device_create();
            device->platform_id = platforms[p];
            device->device_id_cl = devices[d];
            oskar_device_get_info_cl(device);

            // Check if double precision is required.
            if (oskar_device_require_double() && !device->supports_double)
            {
                oskar_device_free(device);
                continue; // Next device.
            }

            // Check device vendor name if specified.
            if (vendors.size() > 0)
            {
                int match = 0;
                std::string vd(device->vendor);
                for (size_t k = 0; k < vd.size(); ++k) vd[k] = toupper(vd[k]);
                for (size_t k = 0; k < vendors.size(); ++k)
                {
                    if (strstr(vd.c_str(), vendors[k].c_str()))
                    {
                        match = 1;
                        break;
                    }
                }
                if (!match)
                {
                    oskar_device_free(device);
                    continue; // Next device.
                }
            }

            // Add to device list.
            device->index = *device_count;
            (*device_count)++;
            device_list = (oskar_Device**) realloc(device_list,
                    *device_count * sizeof(oskar_Device*));
            device_list[*device_count - 1] = device;
        }
    }
#else
    (void) device_type;
    (void) device_vendor;
#endif
    return device_list;
}
