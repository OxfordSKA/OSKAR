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

#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "utility/oskar_cl_registrar.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_thread.h"

using std::map;
using std::string;
using std::vector;

// Private
void oskar_cl_ensure(int no_init);

namespace oskar {

string find_replace(string subject, const string& search, const string& replace)
{
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

struct CLGlobal
{
    unsigned int current_device;

    struct CLDevice
    {
        string name;
        string cl_version;
        string driver_version;
#ifdef OSKAR_HAVE_OPENCL
        cl_device_id id;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        map<string, cl_kernel> kernel;
#endif

        CLDevice()
        {
            name = string();
            cl_version = string();
            driver_version = string();
#ifdef OSKAR_HAVE_OPENCL
            id = 0;
            context = 0;
            queue = 0;
            program = 0;
#endif
        }

        ~CLDevice()
        {
#ifdef OSKAR_HAVE_OPENCL
            if (queue)
            {
                clFlush(queue);
                clFinish(queue);
            }
            for (map<string, cl_kernel>::iterator i = kernel.begin();
                    i != kernel.end(); ++i)
            {
                clReleaseKernel(i->second);
            }
            if (queue) clReleaseCommandQueue(queue);
            if (program) clReleaseProgram(program);
            if (context) clReleaseContext(context);
#endif
        }
    };

    // Data stored per device.
    vector<CLDevice*> device;

    CLGlobal() : current_device(0) {}

    ~CLGlobal()
    {
        clear();
    }

    void clear()
    {
        for (size_t i = 0; i < device.size(); ++i)
            delete device[i];
        device.clear();
    }
};

}


using namespace oskar;

// Thread-local OpenCL data.
// This cannot be a non-pointer type.
static
#ifdef OSKAR_OS_WIN
__declspec(thread)
#else
__thread
#endif
CLGlobal* oskar_cl_ = 0;

// Global pointers to all thread-local objects.
static vector<oskar::CLGlobal*> oskar_cl_all_;

struct LocalMutex
{
    oskar_Mutex* m;
    LocalMutex()
    {
        this->m = oskar_mutex_create();
    }
    ~LocalMutex()
    {
        oskar_mutex_free(this->m);
    }
    void lock()
    {
        oskar_mutex_lock(this->m);
    }
    void unlock()
    {
        oskar_mutex_unlock(this->m);
    }
};
static LocalMutex mutex;

void oskar_cl_ensure(int no_init)
{
    if (!oskar_cl_)
    {
        // Allocate structure and store a pointer to it so it can be
        // accessed by oskar_cl_free().
        oskar_cl_ = new oskar::CLGlobal();
        mutex.lock();
        oskar_cl_all_.push_back(oskar_cl_);
        mutex.unlock();
    }
    if (oskar_cl_->device.size() == 0 && !no_init)
        oskar_cl_init(NULL, NULL);
}

void oskar_cl_free(void)
{
    mutex.lock();
    for (size_t i = 0; i < oskar_cl_all_.size(); ++i)
        delete oskar_cl_all_[i];
    oskar_cl_all_.clear();
    mutex.unlock();
}

void oskar_cl_init(const char* device_type, const char* device_vendor)
{
#ifdef OSKAR_HAVE_OPENCL
    cl_uint num_platforms = 0;
    cl_device_type dev_type = CL_DEVICE_TYPE_ALL;
    cl_int error = 0;
    oskar_cl_ensure(1);

    // Get environment variables if parameters not given.
    if (!device_type || strlen(device_type) == 0)
        device_type = getenv("OSKAR_CL_DEVICE_TYPE");
    if (!device_vendor || strlen(device_vendor) == 0)
        device_vendor = getenv("OSKAR_CL_DEVICE_VENDOR");

    // Get the selected vendor(s).
    vector<string> vendors;
    if (device_vendor && strlen(device_vendor) > 0)
    {
        string delims = "|,";
        string vendor_ = string(device_vendor);
        for (size_t k = 0; k < vendor_.size(); ++k)
            vendor_[k] = toupper(vendor_[k]);
        size_t start = vendor_.find_first_not_of(delims), end = 0;
        while ((end = vendor_.find_first_of(delims, start)) != string::npos)
        {
            vendors.push_back(vendor_.substr(start, end - start));
            start = vendor_.find_first_not_of(delims, end);
        }
        if (start != string::npos)
            vendors.push_back(vendor_.substr(start));
    }

    // Set the device type to match.
    if (device_type)
    {
        switch (toupper(device_type[0]))
        {
        case 'A': dev_type = CL_DEVICE_TYPE_ACCELERATOR; break;
        case 'C': dev_type = CL_DEVICE_TYPE_CPU; break;
        case 'G': dev_type = CL_DEVICE_TYPE_GPU; break;
        default:  dev_type = CL_DEVICE_TYPE_GPU; break;
        }
    }

    // Clear any existing contexts.
    oskar_cl_->clear();

    // Get the OpenCL platform IDs.
    // The first OpenCL call doesn't seem to be thread-safe.
    mutex.lock();
    error = clGetPlatformIDs(0, 0, &num_platforms);
    mutex.unlock();
    if (num_platforms == 0)
    {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return;
    }
    if (error != CL_SUCCESS)
    {
        fprintf(stderr, "clGetPlatformIDs error (%d).\n", error);
        return;
    }
    vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, &platforms[0], 0);

    // Loop over platforms, starting from the last.
    for (int i = (int)num_platforms - 1; i >= 0; i--)
    {
        // Get all device IDs on the platform that match the device type.
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], dev_type, 0, 0, &num_devices);
        if (num_devices == 0) continue;
        vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[i], dev_type, num_devices, &devices[0], 0);

        // Loop over devices.
        for (int j = 0; j < (int)num_devices; j++)
        {
            char* t = 0;
            size_t len = 0;

            // Get device vendor name.
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, 0, &len);
            t = (char*) realloc(t, len);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, len, t, 0);
            //printf("  Device: %d Vendor: %s\n", j, t);

            // Check device vendor name if specified.
            if (vendors.size() > 0)
            {
                int match = 0;
                for (size_t k = 0; k < len; ++k) t[k] = toupper(t[k]);
                for (size_t k = 0; k < vendors.size(); ++k)
                {
                    if (strstr(t, vendors[k].c_str()))
                    {
                        match = 1;
                        break;
                    }
                }
                if (!match) continue; // Next device.
            }

            // Store the device ID, name, device and driver versions.
            oskar_cl_->device.push_back(new CLGlobal::CLDevice());
            CLGlobal::CLDevice* device = oskar_cl_->device.back();
            device->id = devices[j];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, 0, &len);
            t = (char*) realloc(t, len);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, len, t, 0);
            device->name = string(t);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, 0, &len);
            t = (char*) realloc(t, len);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, len, t, 0);
            device->cl_version = string(t);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, 0, &len);
            t = (char*) realloc(t, len);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, len, t, 0);
            device->driver_version = string(t);

            // Create an OpenCL context for the device.
            cl_context_properties props[] =
            {
                CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[i],
                0
            };
            device->context = clCreateContext(props, 1,
                    &devices[j], 0, 0, &error);
            if (error != CL_SUCCESS)
            {
                fprintf(stderr, "clCreateContext error (%d).\n", error);
                break;
            }

            // Check if device supports double precision.
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, 0, &len);
            t = (char*) realloc(t, len);
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, len, t, 0);
            int supports_double = strstr(t, "cl_khr_fp64") ? 1 : 0;

            // Create an OpenCL command queue.
            device->queue = clCreateCommandQueue(device->context, devices[j],
                    0, &error);
            if (error != CL_SUCCESS)
            {
                fprintf(stderr, "clCreateCommandQueue error (%d).\n", error);
                break;
            }

            // Create OpenCL program from kernel sources and build it.
            vector<const char*>& src = CLRegistrar::sources();
            vector<string> sources;
            vector<const char*> source_ptr;
            for (vector<const char*>::iterator k = src.begin();
                    k != src.end(); ++k)
            {
                if (!supports_double && strstr(*k, "double")) continue;
                if (strstr(*k, "REAL"))
                {
                    string s(*k);
                    sources.push_back(find_replace(s, "REAL", "float"));
                    if (supports_double)
                    {
                        s.insert(0, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
                        sources.push_back(find_replace(s, "REAL", "double"));
                    }
                }
                else
                    sources.push_back(string(*k));
            }
            if (sources.empty()) continue;
            for (size_t k = 0; k < sources.size(); ++k)
                source_ptr.push_back(sources[k].c_str());
            device->program = clCreateProgramWithSource(device->context,
                    (cl_uint) source_ptr.size(), &source_ptr[0], 0, &error);
            if (error != CL_SUCCESS)
            {
                fprintf(stderr,
                        "clCreateProgramWithSource error (%d).\n", error);
                break;
            }
            error = clBuildProgram(device->program, 0, 0,
                    "-cl-no-signed-zeros", 0, 0);
            if (error != CL_SUCCESS)
            {
                clGetProgramBuildInfo(device->program,
                        devices[j], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
                t = (char*) realloc(t, len);
                clGetProgramBuildInfo(device->program,
                        devices[j], CL_PROGRAM_BUILD_LOG, len, t, NULL);
                fprintf(stderr, "clBuildProgram error (%d):\n\n%s\n",
                        error, t);
                break;
            }

            // Get all the kernels from the program.
            cl_uint num_kernels = 0;
            error = clCreateKernelsInProgram(device->program,
                    0, NULL, &num_kernels);
            if (error != CL_SUCCESS)
            {
                fprintf(stderr,
                        "clCreateKernelsInProgram error (%d).\n", error);
                break;
            }
            if (num_kernels == 0) continue;
            vector<cl_kernel> kernels(num_kernels);
            error = clCreateKernelsInProgram(device->program,
                    num_kernels, &kernels[0], NULL);
            if (error != CL_SUCCESS)
            {
                fprintf(stderr,
                        "clCreateKernelsInProgram error (%d).\n", error);
                break;
            }
            for (cl_uint k = 0; k < num_kernels; ++k)
            {
                clGetKernelInfo(kernels[k], CL_KERNEL_FUNCTION_NAME,
                        0, NULL, &len);
                t = (char*) realloc(t, len);
                clGetKernelInfo(kernels[k], CL_KERNEL_FUNCTION_NAME,
                        len, t, NULL);
                device->kernel[string(t)] = kernels[k];
            }
            free(t);
        }
    }
#else
    (void) device_type;
    (void) device_vendor;
#endif
}

#ifdef OSKAR_HAVE_OPENCL

cl_command_queue oskar_cl_command_queue(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ? oskar_cl_->device[i]->queue : 0;
}

cl_context oskar_cl_context(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ? oskar_cl_->device[i]->context : 0;
}

cl_device_id oskar_cl_device_id(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ? oskar_cl_->device[i]->id : 0;
}

cl_kernel oskar_cl_kernel(const char* name)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    if (i < oskar_cl_->device.size())
    {
        CLGlobal::CLDevice* device = oskar_cl_->device[i];
        map<string, cl_kernel>::iterator j = device->kernel.find(name);
        if (j != device->kernel.end())
            return j->second;
    }
    return 0;
}

#endif

const char* oskar_cl_device_cl_version(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ?
            oskar_cl_->device[i]->cl_version.c_str() : 0;
}

const char* oskar_cl_device_driver_version(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ?
            oskar_cl_->device[i]->driver_version.c_str() : 0;
}

const char* oskar_cl_device_name(void)
{
    oskar_cl_ensure(0);
    unsigned int i = oskar_cl_->current_device;
    return i < oskar_cl_->device.size() ?
            oskar_cl_->device[i]->name.c_str() : 0;
}

unsigned int oskar_cl_get_device(void)
{
    oskar_cl_ensure(0);
    return oskar_cl_->current_device;
}

unsigned int oskar_cl_num_devices(void)
{
    oskar_cl_ensure(0);
    return (unsigned int) oskar_cl_->device.size();
}

void oskar_cl_set_device(unsigned int device, int* status)
{
    oskar_cl_ensure(0);
    if (device >= oskar_cl_->device.size())
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    oskar_cl_->current_device = device;
}
