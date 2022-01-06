/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "utility/oskar_device.h"
#include "utility/private_device.h"

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "binary/oskar_binary.h"
#include "binary/oskar_crc.h"
#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_cl_registrar.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_lock_file.h"
#include "utility/oskar_thread.h"

#ifdef OSKAR_OS_WIN
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

struct oskar_DeviceKernels
{
    std::string src;
#ifdef OSKAR_HAVE_OPENCL
    std::map<std::string, cl_kernel> kernel;
#endif
    virtual ~oskar_DeviceKernels()
    {
#ifdef OSKAR_HAVE_OPENCL
        for (std::map<std::string, cl_kernel>::iterator i = kernel.begin();
                i != kernel.end(); ++i)
        {
            clReleaseKernel(i->second);
        }
#endif
    }
};

static void oskar_device_set_up_cl(oskar_Device* device);

static THREAD_LOCAL unsigned int current_platform_ = 0;
static THREAD_LOCAL unsigned int current_device_ = 0;
static std::vector<oskar_Device*> cl_devices_;
static std::map<std::string, const void*> cuda_kernels_;
static int require_double_ = 1; // Set if double precision is required.

struct oskar_LocalMutex
{
    oskar_Mutex* m;
    oskar_LocalMutex()  { this->m = oskar_mutex_create(); }
    ~oskar_LocalMutex() { oskar_mutex_free(this->m); }
    void lock() const   { oskar_mutex_lock(this->m); }
    void unlock() const { oskar_mutex_unlock(this->m); }
};
static oskar_LocalMutex mutex_; // NOLINT: This constructor will not throw.

void oskar_device_check_error_cuda(int* status)
{
    if (*status) return;
#ifdef OSKAR_HAVE_CUDA
    *status = (int) cudaPeekAtLastError();
#endif
}

void oskar_device_check_local_size(int location, unsigned int dim,
        size_t local_size[3])
{
    size_t max_size = 1;
    if (location == OSKAR_GPU)
    {
        max_size = (dim == 0 || dim == 1) ? 1024 : 64;
    }
    else if (location & OSKAR_CL)
    {
        if (oskar_device_is_cpu(location))
        {
            max_size = 128;
        }
        else if (current_device_ < cl_devices_.size() && dim < 3)
        {
            max_size = cl_devices_[current_device_]->max_local_size[dim];
        }
    }
    if (local_size[dim] > max_size) local_size[dim] = max_size;
}

struct _cl_context* oskar_device_context_cl(void)
{
    if (cl_devices_.size() == 0) oskar_device_init_cl();
    unsigned int i = current_device_;
    return i < cl_devices_.size() ? cl_devices_[i]->context : 0;
}

oskar_Device* oskar_device_create(void)
{
    return (oskar_Device*) calloc(1, sizeof(oskar_Device));
}

const oskar_Device* oskar_device_cl(int id)
{
    if (cl_devices_.size() == 0) oskar_device_init_cl();
    return id < (int)cl_devices_.size() ? cl_devices_[id] : 0;
}

void oskar_device_free(oskar_Device* device)
{
    if (!device) return;
    free(device->name);
    free(device->vendor);
    free(device->cl_version);
    free(device->cl_driver_version);
    if (device->kern) delete device->kern;
#ifdef OSKAR_HAVE_OPENCL
    if (device->default_queue)
    {
        clFlush(device->default_queue);
        clFinish(device->default_queue);
        clReleaseCommandQueue(device->default_queue);
    }
    if (device->program) clReleaseProgram(device->program);
    if (device->context) clReleaseContext(device->context);
#endif
    free(device);
}

size_t oskar_device_global_size(size_t num, size_t local_size)
{
    return ((num + local_size - 1) / local_size) * local_size;
}

void oskar_device_init_cl(void)
{
    int num_devices = 0;
    oskar_log_section(0, 'S', "OpenCL device set-up");
    oskar_Device** devices = oskar_device_create_list(OSKAR_CL, &num_devices);
    for (int i = 0; i < (int)cl_devices_.size(); ++i)
    {
        oskar_device_free(cl_devices_[i]);
    }
    cl_devices_.clear();
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_device_set_up_cl(devices[i]);
        cl_devices_.push_back(devices[i]);
    }
}

int oskar_device_is_cpu(int location)
{
    if (location == OSKAR_GPU) return 0;
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        return current_device_ < cl_devices_.size() ?
                (cl_devices_[current_device_]->device_type == 'C') : 0;
    }
    return 1;
}

int oskar_device_is_gpu(int location)
{
    if (location == OSKAR_GPU) return 1;
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        return current_device_ < cl_devices_.size() ?
                (cl_devices_[current_device_]->device_type == 'G') : 0;
    }
    return 0;
}

int oskar_device_is_nv(int location)
{
    if (location == OSKAR_GPU) return 1;
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        return current_device_ < cl_devices_.size() ?
                cl_devices_[current_device_]->is_nv : 0;
    }
    return 0;
}

void oskar_device_launch_kernel(const char* name, int location,
        int num_dims, size_t local_size[3], size_t global_size[3],
        size_t num_args, const oskar_Arg* arg,
        size_t num_local_args, const size_t* arg_size_local, int* status)
{
    if (*status) return;
    if (!name)
    {
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        return;
    }
    if (local_size[0] == 0) local_size[0] = 1;
    if (local_size[1] == 0) local_size[1] = 1;
    if (local_size[2] == 0) local_size[2] = 1;
    if (global_size[0] == 0) global_size[0] = 1;
    if (global_size[1] == 0) global_size[1] = 1;
    if (global_size[2] == 0) global_size[2] = 1;
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        size_t j = 0, shared_mem = 0;
        dim3 num_threads, num_blocks;
        void* arg_[40];
        (void) num_dims;
        if (num_args > (sizeof(arg_) / sizeof(void*)))
        {
            *status = OSKAR_ERR_OUT_OF_RANGE;
            return;
        }
        num_threads.x = (unsigned int) local_size[0];
        num_threads.y = (unsigned int) local_size[1];
        num_threads.z = (unsigned int) local_size[2];
        num_blocks.x  = (unsigned int) (global_size[0] / local_size[0]);
        num_blocks.y  = (unsigned int) (global_size[1] / local_size[1]);
        num_blocks.z  = (unsigned int) (global_size[2] / local_size[2]);
        for (j = 0; j < num_args; ++j) arg_[j] = const_cast<void*>(arg[j].ptr);
        for (j = 0; j < num_local_args; ++j) shared_mem += arg_size_local[j];
        mutex_.lock();
        if (cuda_kernels_.empty())
        {
            const oskar::CudaKernelRegistrar::List& kernels =
                    oskar::CudaKernelRegistrar::kernels();
            for (int i = 0; i < kernels.size(); ++i)
            {
                std::string key = std::string(kernels[i].first);
                cuda_kernels_.insert(make_pair(key, kernels[i].second));
            }
        }
        std::map<std::string, const void*>::iterator iter =
                cuda_kernels_.find(std::string(name));
        mutex_.unlock();
        if (iter != cuda_kernels_.end())
        {
            *status = (int) cudaLaunchKernel(iter->second,
                    num_blocks, num_threads, arg_, shared_mem, 0);
            if (*status != 0)
            {
                oskar_log_error(0,
                        "Kernel '%s' launch failure (CUDA error %d: %s).",
                        name, *status,
                        cudaGetErrorString((cudaError_t)(*status)));
            }
            if (*status == cudaErrorInvalidConfiguration)
            {
                oskar_log_error(0,
                        "Number of threads/blocks: (%u, %u, %u)/(%u, %u, %u), "
                        "shared memory: %zu",
                        num_threads.x, num_threads.y, num_threads.z,
                        num_blocks.x, num_blocks.y, num_blocks.z, shared_mem);
            }
        }
        else
        {
            oskar_log_error(0, "Kernel '%s' has not been registered.", name);
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        size_t i = 0, j = 0, work_group_size = 0;
        cl_int error = 0;
        cl_kernel k = 0;
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        if (current_device_ >= cl_devices_.size()) return;
        oskar_Device* device = cl_devices_[current_device_];
        std::map<std::string, cl_kernel>::iterator iter =
                device->kern->kernel.find(name);
        if (iter != device->kern->kernel.end()) k = iter->second;
        if (!k)
        {
            oskar_log_error(0, "Kernel '%s' has not been registered.", name);
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }
        for (i = 0; i < num_args; ++i)
        {
            error |= clSetKernelArg(k, (cl_uint) i, arg[i].size, arg[i].ptr);
        }
        if (!oskar_device_is_cpu(location))
        {
            for (j = 0; j < num_local_args; ++j)
            {
                size_t t = arg_size_local[j];
                if (t == 0) t = 8;
                error |= clSetKernelArg(k, (cl_uint) (i + j), t, 0);
            }
        }
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return;
        }
        error = clEnqueueNDRangeKernel(oskar_device_queue_cl(), k,
                (cl_uint) num_dims, NULL, global_size, local_size,
                0, NULL, NULL);
        if (error != CL_SUCCESS)
        {
            oskar_log_error(0, "Kernel '%s' launch failure (OpenCL error %d).",
                    name, error);
            if (error == CL_INVALID_WORK_GROUP_SIZE)
            {
                clGetKernelWorkGroupInfo(k, device->device_id_cl,
                        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                        &work_group_size, 0);
                oskar_log_message(0, 'M', 1, "Local size is: (%zu, %zu, %zu)",
                        local_size[0], local_size[1], local_size[2]);
                oskar_log_message(0, 'M', 1, "Allowed size for '%s' is %zu",
                        name, work_group_size);
            }
            *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

char* oskar_device_name(int location, int id)
{
    char* name = 0;
    if (location == OSKAR_GPU)
    {
        oskar_Device* device = oskar_device_create();
        device->index = id;
        oskar_device_get_info_cuda(device);
        const size_t buffer_size = 1 + strlen(device->name);
        name = (char*) calloc(buffer_size, 1);
        if (name) memcpy(name, device->name, buffer_size);
        oskar_device_free(device);
    }
    else if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        if (id >= (int) cl_devices_.size()) return name;
        const size_t buffer_size = 1 + strlen(cl_devices_[id]->name);
        name = (char*) calloc(buffer_size, 1);
        if (name) memcpy(name, cl_devices_[id]->name, buffer_size);
    }
    return name;
}

struct _cl_command_queue* oskar_device_queue_cl(void)
{
    if (cl_devices_.size() == 0) oskar_device_init_cl();
    unsigned int i = current_device_;
    return i < cl_devices_.size() ? cl_devices_[i]->default_queue : 0;
}

int oskar_device_require_double(void)
{
    return require_double_;
}

void oskar_device_reset_all(void)
{
#ifdef OSKAR_HAVE_CUDA
    int num = 0;
    if (cudaGetDeviceCount(&num) != cudaSuccess) num = 0;
    for (int i = 0; i < num; ++i)
    {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
#endif
    mutex_.lock();
    for (size_t i = 0; i < cl_devices_.size(); ++i)
    {
        oskar_device_free(cl_devices_[i]);
    }
    cl_devices_.clear();
    current_device_ = 0;
    mutex_.unlock();
}

void oskar_device_set(int location, int id, int* status)
{
    if (*status || id < 0) return;
#ifdef OSKAR_HAVE_CUDA
    if (location == OSKAR_GPU)
    {
        *status = (int) cudaSetDevice(id);
        current_device_ = id;
        current_platform_ = location;
    }
    else
#endif
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        if (id >= (int) cl_devices_.size())
        {
            *status = OSKAR_ERR_OUT_OF_RANGE;
            return;
        }
        current_device_ = id;
        current_platform_ = location;
    }
}

void oskar_device_set_require_double_precision(int flag)
{
    require_double_ = flag;
}

int oskar_device_supports_atomic64(int location)
{
    if (location == OSKAR_GPU) return 1;
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        return current_device_ < cl_devices_.size() ?
                (cl_devices_[current_device_]->supports_atomic64) : 0;
    }
    return 0;
}

int oskar_device_supports_double(int location)
{
    if (location == OSKAR_GPU) return 1;
    if (location & OSKAR_CL)
    {
        if (cl_devices_.size() == 0) oskar_device_init_cl();
        return current_device_ < cl_devices_.size() ?
                (cl_devices_[current_device_]->supports_double) : 0;
    }
    return 0;
}

/****************************************************************************/
/* Private functions */
/****************************************************************************/

static char* oskar_device_cache_path(const oskar_Device* device,
        int* cache_exists)
{
    std::string t = std::string(".oskar_cache_") + std::string(device->name);
    for (size_t i = 0; i < t.length(); ++i)
    {
        if (t[i] == ' ' || t[i] == '@' || t[i] == '(' || t[i] == ')')
        {
            t[i] = '_';
        }
    }
    return oskar_dir_get_home_path(t.c_str(), cache_exists);
}

static unsigned long int oskar_device_crc(const oskar_Device* device,
        const char* program_source, size_t program_length)
{
    unsigned long int crc = 0;
    oskar_CRC* data = oskar_crc_create(OSKAR_CRC_32C);
    crc = oskar_crc_update(data, crc, device->vendor, strlen(device->vendor));
    crc = oskar_crc_update(data, crc, device->name, strlen(device->name));
    crc = oskar_crc_update(data, crc,
            device->cl_driver_version, strlen(device->cl_driver_version));
    crc = oskar_crc_update(data, crc, program_source, program_length);
    oskar_crc_free(data);
    return crc;
}

static unsigned char* oskar_device_binary_load_cl(
        const oskar_Device* device, int* program_binary_size)
{
    int cache_exists = 0;
    unsigned char* program_binary = 0;
    char* cache_name = oskar_device_cache_path(device, &cache_exists);
    if (cache_exists)
    {
        int file_program_source_len = 0, program_source_len = 0, status = 0;
        unsigned long int file_program_crc = 0, program_crc = 0;
        program_source_len = (int) device->kern->src.length();
        program_crc = oskar_device_crc(device,
                device->kern->src.c_str(), program_source_len);
        oskar_log_message(0, 'S', 0, "Loading OpenCL program for %s",
                device->name);
        oskar_Binary* file = oskar_binary_create(cache_name, 'r', &status);
        oskar_binary_read_ext(file, OSKAR_INT, device->name,
                "PROGRAM_CRC", 0, sizeof(unsigned long int),
                &file_program_crc, &status);
        oskar_binary_read_ext(file, OSKAR_INT, device->name,
                "PROGRAM_SOURCE_LENGTH", 0, sizeof(int),
                &file_program_source_len, &status);
        oskar_binary_read_ext(file, OSKAR_INT, device->name,
                "PROGRAM_BINARY_SIZE", 0, sizeof(int),
                program_binary_size, &status);
        if (file_program_crc == program_crc &&
                file_program_source_len == program_source_len &&
                *program_binary_size > 0)
        {
            program_binary = (unsigned char*) malloc(*program_binary_size);
            oskar_binary_read_ext(file, OSKAR_CHAR, device->name,
                    "PROGRAM_BINARY", 0, *program_binary_size,
                    program_binary, &status);
        }
        oskar_binary_free(file);
    }
    free(cache_name);
    return program_binary;
}

static void oskar_device_binary_save_cl(const oskar_Device* device)
{
#ifdef OSKAR_HAVE_OPENCL
    int status = 0;
    char* cache_name = oskar_device_cache_path(device, 0);
    std::string lock_name = std::string(cache_name) + std::string(".lock");
    if (oskar_lock_file(lock_name.c_str()))
    {
        oskar_Binary* file = oskar_binary_create(cache_name, 'w', &status);
        if (!file || status)
        {
            oskar_binary_free(file);
            remove(lock_name.c_str());
            free(cache_name);
            return;
        }
        cl_uint num_devices = 0;
        clGetProgramInfo(device->program, CL_PROGRAM_NUM_DEVICES,
                sizeof(cl_uint), &num_devices, 0);
        if (num_devices != 1) return;
        size_t* binary_sizes = (size_t*) malloc(num_devices * sizeof(size_t));
        clGetProgramInfo(device->program, CL_PROGRAM_BINARY_SIZES,
                num_devices * sizeof(size_t), binary_sizes, 0);
        unsigned char** binaries =
                (unsigned char**) malloc(num_devices * sizeof(unsigned char*));
        for (cl_uint i = 0; i < num_devices; ++i)
        {
            binaries[i] = (unsigned char*) malloc(binary_sizes[i]);
        }
        clGetProgramInfo(device->program, CL_PROGRAM_BINARIES,
                num_devices * sizeof(unsigned char*), binaries, 0);
        const int program_source_len = (int) device->kern->src.length();
        const int program_binary_size = (int) binary_sizes[0];
        const unsigned long int program_crc = oskar_device_crc(device,
                device->kern->src.c_str(), (size_t) program_source_len);
        oskar_binary_write_ext(file, OSKAR_INT, device->name,
                "PROGRAM_CRC", 0, sizeof(unsigned long int),
                &program_crc, &status);
        oskar_binary_write_ext(file, OSKAR_INT, device->name,
                "PROGRAM_SOURCE_LENGTH", 0, sizeof(int),
                &program_source_len, &status);
        oskar_binary_write_ext(file, OSKAR_INT, device->name,
                "PROGRAM_BINARY_SIZE", 0, sizeof(int),
                &program_binary_size, &status);
        oskar_binary_write_ext(file, OSKAR_CHAR, device->name,
                "PROGRAM_BINARY", 0, binary_sizes[0],
                binaries[0], &status);
        for (cl_uint i = 0; i < num_devices; ++i) free(binaries[i]);
        free(binaries);
        free(binary_sizes);
        oskar_binary_free(file);
        remove(lock_name.c_str());
    }
    free(cache_name);
#else
    (void) device;
#endif
}

static std::string find_replace(std::string subject,
        const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}

static void oskar_device_set_up_cl(oskar_Device* device)
{
    int error = 0;
    if (!device->init)
    {
        oskar_device_get_info_cl(device);
    }
    if (!device->kern)
    {
        device->kern = new oskar_DeviceKernels;
    }
    std::vector<const char*>& sv = oskar::CLRegistrar::sources();
    std::deque<std::string> headers, sources;
    enum {HEADER, SOURCE};
    for (std::vector<const char*>::iterator i = sv.begin(); i != sv.end(); ++i)
    {
        std::string s(*i);
        int fragment_type = SOURCE;
        if (strstr(s.c_str(), "typedef") || strstr(s.c_str(), "#define"))
        {
            fragment_type = HEADER;
        }
        if (fragment_type == SOURCE)
        {
            if ((strstr(s.c_str(), "_CPU") && device->device_type != 'C') ||
                    (strstr(s.c_str(), "_GPU") && device->device_type != 'G'))
            {
                continue;
            }
        }
        if (strstr(s.c_str(), "Real"))
        {
            std::string orig(s);
            s = find_replace(orig, "Real", "float");
            if (device->supports_double)
            {
                s.append(find_replace(orig, "Real", "double"));
            }
        }
        if (fragment_type == HEADER)
        {
            headers.push_back(s);
        }
        else
        {
            sources.push_back(s);
        }
    }
    if (device->supports_atomic64)
    {
        headers.push_front("#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n");
    }
    if (device->supports_atomic32)
    {
        headers.push_front("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n");
    }
    if (device->supports_double)
    {
        headers.push_front("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    }
    if (device->supports_double && device->supports_atomic64)
    {
        headers.push_front("#define PREFER_DOUBLE double\n");
    }
    else
    {
        headers.push_front("#define PREFER_DOUBLE float\n");
    }
    std::string& src = device->kern->src;
    src.clear();
    for (size_t i = 0; i < headers.size(); ++i) src.append(headers[i]);
    for (size_t i = 0; i < sources.size(); ++i) src.append(sources[i]);
    //printf("%s\n", src.c_str());

    // Use cached program binary if available.
    int program_binary_size = 0, used_binary = 0;
    unsigned char* program_binary = oskar_device_binary_load_cl(device,
            &program_binary_size);

#ifdef OSKAR_HAVE_OPENCL
    // Create OpenCL context and program.
    const char* func = 0;
    cl_context_properties props[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties) device->platform_id, 0
    };
    device->context = clCreateContext(props,
            1, &device->device_id_cl, 0, 0, &error);
    if (error != CL_SUCCESS)
    {
        func = "clCreateContext";
    }
    else
    {
        cl_int binary_status = CL_SUCCESS;
        if (program_binary)
        {
            const size_t len[] = { (size_t)program_binary_size };
            const unsigned char* bin[] = { program_binary };
            device->program = clCreateProgramWithBinary(device->context, 1,
                    &device->device_id_cl, len, bin, &binary_status, &error);
            if (error != CL_SUCCESS)
            {
                func = "clCreateProgramWithBinary";
            }
            else if (binary_status == CL_SUCCESS)
            {
                used_binary = 1;
            }
        }
        if (!program_binary || binary_status != CL_SUCCESS)
        {
            const char* src_ptr[] = { src.c_str() };
            used_binary = 0;
            oskar_log_message(0, 'S', 0,
                    "Building OpenCL program for %s, please wait...",
                    device->name);
            oskar_log_message(0, 'S', 1, "Required on first run of a new "
                    "version, or if GPU drivers are updated.");
            device->program = clCreateProgramWithSource(device->context, 1,
                    src_ptr, 0, &error);
            if (error != CL_SUCCESS) func = "clCreateProgramWithSource";
        }
    }

    // Build program and get kernels from it.
    cl_uint num_kernels = 0;
    std::vector<cl_kernel> kernels;
    if (error == CL_SUCCESS)
    {
        if ((error = clBuildProgram(device->program, 0, 0,
                "-cl-mad-enable -cl-no-signed-zeros ", 0, 0)) == CL_SUCCESS)
        {
            if ((error = clCreateKernelsInProgram(device->program,
                    0, NULL, &num_kernels)) != CL_SUCCESS)
            {
                func = "clCreateKernelsInProgram";
            }
            kernels.resize(num_kernels);
            if (error == CL_SUCCESS && num_kernels > 0)
            {
                if ((error = clCreateKernelsInProgram(device->program,
                        num_kernels, &kernels[0], NULL)) != CL_SUCCESS)
                {
                    func = "clCreateKernelsInProgram";
                }
            }
        }
        else
        {
            size_t len = 0;
            func = "clBuildProgram";
            clGetProgramBuildInfo(device->program, device->device_id_cl,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            char* build_log = (char*) malloc(len);
            clGetProgramBuildInfo(device->program, device->device_id_cl,
                    CL_PROGRAM_BUILD_LOG, len, build_log, NULL);
            fprintf(stderr, "%s error (code %d):\n\n%s\n",
                    func, error, build_log);
            free(build_log);
        }
    }
    if (error == CL_SUCCESS)
    {
        char* t = 0;
        size_t len = 0;
        for (cl_uint k = 0; k < num_kernels; ++k)
        {
            clGetKernelInfo(kernels[k], CL_KERNEL_FUNCTION_NAME, 0, NULL, &len);
            t = (char*) realloc(t, len);
            clGetKernelInfo(kernels[k], CL_KERNEL_FUNCTION_NAME, len, t, NULL);
            device->kern->kernel[std::string(t)] = kernels[k];
        }
        free(t);

        // Create a default command queue.
        device->default_queue = clCreateCommandQueue(device->context,
                device->device_id_cl, CL_QUEUE_PROFILING_ENABLE, &error);
        if (error != CL_SUCCESS) func = "clCreateCommandQueue";
    }
    if (error || func) oskar_log_error(0, "%s error (%d).", func, error);
#endif
    free(program_binary);

    // Save program binary to file if it was built from source.
    if (!used_binary && !error)
    {
        oskar_device_binary_save_cl(device);
    }
}
