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
    cudaError_t* error = (cudaError_t*) mxGetPr(output[0]);
    *error = cudaSuccess;
    *error = cudaPeekAtLastError();
    *error = (cudaError_t)3;
    const char* error_string = cudaGetErrorString(*error);
    if (*error != cudaSuccess)
        mexPrintf("CUDA ERROR[%i]: %s.\n", (int)*error, error_string);
}
