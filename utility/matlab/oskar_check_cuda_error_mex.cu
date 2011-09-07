#include <mex.h>
#include <string>

// Interface function - can call anything from here...
//
// nlhs = The number of left-hand arguments, or the size of the plhs array.
// plhs = An array of left-hand output arguments.
// nrhs = The number of right-hand arguments, or the size of the prhs array.
// prhs = An array of left-hand output arguments.
void mexFunction(int /*num_outputs*/, mxArray** output, int num_inputs,
        const mxArray** input)
{
    // Check for proper number of arguments.
    if (num_inputs != 0)
        mexErrMsgTxt("ERROR: Not expecting any arguments.");

    mwSize n_dims = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxINT32_CLASS, mxREAL);

    cudaError_t err = (cudaError_t) mxGetScalar(output[0]);
    err = cudaSuccess;

    err = cudaPeekAtLastError();

    // http://www.txcorp.com/products/GPULib/idl_docs/errorcodes.html
    std::string s_err;
    switch (err)
    {
        case cudaErrorMissingConfiguration:
            s_err = "Missing configuration error";
            break;
        case cudaErrorMemoryAllocation:
            s_err = "Memory allocation error";
            break;
        case cudaErrorInitializationError:
            s_err = "Initialization error";
            break;
        case cudaErrorLaunchFailure:
            s_err = "Launch failure";
            break;
        case cudaErrorPriorLaunchFailure:
            s_err = "Prior launch failure";
            break;
        case cudaErrorLaunchTimeout:
            s_err = "Launch timeout error";
            break;
        default:
            s_err = "unspecified error - consult a the CUDA manual";
            break;
    };

    if (err != cudaSuccess)
        mexPrintf("CUDA ERROR[%i]: %s.\n", (int)err, s_err.c_str());
}
