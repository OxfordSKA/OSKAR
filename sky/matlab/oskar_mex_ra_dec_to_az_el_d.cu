#include <mex.h>

#include "sky/oskar_cuda_ra_dec_to_az_el.h"

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    cudaDeviceReset();

    // Parse Inputs.
    if (num_inputs != 4)
        mexErrMsgTxt("Four inputs required ==> (ra_rad, dec_rad, lst_rad, lat_rad)");

    // Get matlab inputs.
    double ra  = mxGetScalar(input[0]);
    double dec = mxGetScalar(input[1]);
    double lst = mxGetScalar(input[2]);
    double lat = mxGetScalar(input[3]);

    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    output[1] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* az = (double*) mxGetPr(output[0]);
    double* el = (double*) mxGetPr(output[1]);

    int error = (int)cudaSuccess;
    error = oskar_ra_dec_to_az_el_d(ra, dec, lst, lat, az, el);

    if (error != cudaSuccess)
    {
        mexPrintf("****************************************************\n");
        mexPrintf("** CUDA ERROR[%i]: %s.\n", error,
                cudaGetErrorString((cudaError_t)error));
        mexPrintf("****************************************************\n");
        mexErrMsgTxt("** ERROR: oskar_ra_dec_to_az_el_d()");
    }
}
