#include <mex.h>

#include "sky/oskar_mjd_to_last_fast.h"

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // Parse Inputs.
    if (num_inputs != 2)
        mexErrMsgTxt("Two inputs required ==> (mjd (UT1), longitude (radians))");

    // Get matlab inputs.
    double mjd_utc  = mxGetScalar(input[0]);
    double lon_rad  = mxGetScalar(input[1]);

    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* last_rad = (double*) mxGetPr(output[0]);

    *last_rad = oskar_mjd_to_last_fast_d(mjd_utc, lon_rad);
}
