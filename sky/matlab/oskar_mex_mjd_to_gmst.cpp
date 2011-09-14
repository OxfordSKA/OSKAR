#include <mex.h>

#include "sky/oskar_mjd_to_gmst.h"

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // Parse Inputs.
    if (num_inputs != 1)
        mexErrMsgTxt("One input required ==> (mjd (UT1)");

    // Get matlab inputs.
    int mjd_ut1   = mxGetScalar(input[0]);

    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* gmst_rad = (double*) mxGetPr(output[0]);

    *gmst_rad = oskar_mjd_to_gmst_d(mjd_ut1);
}
