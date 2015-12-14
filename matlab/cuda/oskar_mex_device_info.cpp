#ifdef _MSC_VER
    typedef __int32 int32_t;
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
#else
    #include <stdint.h>
#endif

#include "matlab/common/oskar_matlab_common.h"

#include <mex.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <vector>

// Cleanup function called when the mex function is unloaded. (i.e. 'clear mex')
void cleanup(void)
{
    cudaDeviceReset();
}

// Interface function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** /*in*/)
{
    // Check for proper number of arguments.
    if (num_in != 0 || num_out > 1)
    {
        oskar_matlab_usage("[info]", "cuda", "device_info", "",
                "Returns an array of structures containing very basic CUDA "
                "device information for each device found");
    }

    int device_id = 0;
    cudaGetDevice(&device_id);

    mexAtExit(cleanup);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    const char* fields[5] = { "error_code", "error_message", "double_support",
            "mem_free", "mem_total" };
    out[0] = mxCreateStructMatrix(device_count, 1, 5, fields);

    // Populate structure per device
    for (int i = 0; i < device_count; ++i)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();

        // Get the current CUDA error status.
        cudaError_t error = cudaPeekAtLastError();
        mxArray* error_code = mxCreateNumericMatrix(1,1, mxINT32_CLASS, mxREAL);
        *(int*)mxGetData(error_code) = (int)error;
        mxSetField(out[0], i, "error_code", error_code);
        mxArray* error_message = mxCreateString(cudaGetErrorString(error));
        mxSetField(out[0], i, "error_message", error_message);

        // Find if double is supported.
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        mxArray* double_support = mxCreateNumericMatrix(1, 1, mxLOGICAL_CLASS, mxREAL);
        if (device_prop.major >= 2 || device_prop.minor >= 3)
            *mxGetLogicals(double_support) = true;
        else
            *mxGetLogicals(double_support) = false;
        mxSetField(out[0], i, "double_support", double_support);

        // Get memory into and populate mem_free and mem_total fields.
        mxArray* mem_free  = mxCreateNumericMatrix(1,1, mxINT64_CLASS, mxREAL);
        mxArray* mem_total = mxCreateNumericMatrix(1,1, mxINT64_CLASS, mxREAL);
        size_t* free  = (size_t*)mxGetData(mem_free);
        size_t* total = (size_t*)mxGetData(mem_total);
        cudaMemGetInfo(free, total);
        mxSetField(out[0], i, "mem_free", mem_free);
        mxSetField(out[0], i, "mem_total", mem_total);
    }
    cudaSetDevice(device_id);
}
