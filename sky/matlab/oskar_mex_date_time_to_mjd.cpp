#include <mex.h>

#include "sky/oskar_date_time_to_mjd.h"

void mexFunction(int /*num_outputs*/, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    // Parse Inputs.
    if (num_inputs != 4)
        mexErrMsgTxt("Four inputs required ==> (year, month, day, day_fraction)");

    // Get matlab inputs.
    int year   = mxGetScalar(input[0]);
    int month  = mxGetScalar(input[1]);
    int day    = mxGetScalar(input[2]);
    double day_fraction = mxGetScalar(input[3]);

    mwSize n_dims  = 1;
    mwSize dims[1] = {1};
    output[0] = mxCreateNumericArray(n_dims, dims, mxDOUBLE_CLASS, mxREAL);
    double* mjd = (double*) mxGetPr(output[0]);

    *mjd = oskar_date_time_to_mjd(year, month, day, day_fraction);
}
