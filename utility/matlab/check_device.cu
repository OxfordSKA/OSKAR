#ifdef _MSC_VER
    typedef __int32 int32_t;
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
#else
    #include <stdint.h>
#endif

    #include <mex.h>
#include <string.h>
#include <cuda_runtime_api.h>

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check for proper number of arguments.
    if (num_in != 0 || num_out > 2)
    {
        mexErrMsgTxt("Usage: status = check_device()");
    }

    // Get the free and total device memory in bytes.
    cudaDeviceSynchronize();
    size_t mem_free = 0, mem_total = 0;
    cudaMemGetInfo(&mem_free, &mem_total);

    cudaError_t err = cudaPeekAtLastError();
    mxArray* error = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    *((int*)mxGetData(error))  = (int)err;
    mxArray* error_string = mxCreateString(cudaGetErrorString(err));
    mxArray* free  = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxArray* total = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(free))  = (uint64_t)mem_free;
    *((uint64_t*)mxGetData(total)) = (uint64_t)mem_total;

    const char* fields[4] = {"free", "total", "error_code", "error_message"};
    out[0] = mxCreateStructMatrix(1, 1, 4, fields);
    mxSetFieldByNumber(out[0], 0, mxGetFieldNumber(out[0], "free"), free);
    mxSetFieldByNumber(out[0], 0, mxGetFieldNumber(out[0], "total"), total);
    mxSetFieldByNumber(out[0], 0, mxGetFieldNumber(out[0], "error_code"), error);
    mxSetFieldByNumber(out[0], 0, mxGetFieldNumber(out[0], "error_message"), error_string);
}
