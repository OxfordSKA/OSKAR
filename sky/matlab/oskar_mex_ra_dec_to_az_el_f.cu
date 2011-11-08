#include <mex.h>

#include "sky/oskar_ra_dec_to_az_el_cuda.h"

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // Parse Inputs.
    if (num_inputs != 4)
        mexErrMsgTxt("Four inputs required ==> (ra_rad, dec_rad, lst_rad, lat_rad)");

    // Get matlab inputs.
    float ra  = (float) mxGetScalar(input[0]);
    float dec = (float) mxGetScalar(input[1]);
    float lst = (float) mxGetScalar(input[2]);
    float lat = (float) mxGetScalar(input[3]);

    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    output[1] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* az = (double*) mxGetPr(output[0]);
    double* el = (double*) mxGetPr(output[1]);

    float az_temp, el_temp;
    int error = (int)cudaSuccess;
    error = oskar_ra_dec_to_az_el_f(ra, dec, lst, lat, &az_temp, &el_temp);

    *az = az_temp;
    *el = el_temp;

    if (error != cudaSuccess)
    {
        mexPrintf("****************************************************\n");
        mexPrintf("** CUDA ERROR[%i]: %s.\n", error,
                cudaGetErrorString((cudaError_t)error));
        mexPrintf("****************************************************\n");
        mexErrMsgTxt("** ERROR: oskar_ra_dec_to_az_el_f()");
    }
}
