#include <mex.h>
#include <string>
#include <cuda_runtime_api.h>

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


    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    output[1] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* rtn_free  = (double*) mxGetPr(output[0]);
    double* rtn_total = (double*) mxGetPr(output[1]);

     // http://stackoverflow.com/questions/7068280/does-matlab-cause-cuda-to-leak-memory-due-to-cucontext-caching
     size_t free = 0, total = 0;
     cudaMemGetInfo(&free, &total);
     mexPrintf("CUDA device memory: Free %u bytes (%u MB) | Total %u bytes (%u MB). ",
             free, free/(1024 * 1024), total, total / (1024 *1024));
     if( total > 0 )
         mexPrintf("%2.2f%% free.\n", (100.0 * free) / total );
     else
         mexPrintf("\n");

     *rtn_free  = (double)free  / (1024 * 1024);
     *rtn_total = (double)total / (1024 * 1024);
}
