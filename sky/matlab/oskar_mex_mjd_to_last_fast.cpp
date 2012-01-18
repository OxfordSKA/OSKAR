#include <mex.h>

#include "sky/oskar_mjd_to_last_fast.h"

void mexFunction(int /*num_out*/, mxArray** out, int num_in, const mxArray** in)
{
    // Parse Inputs.
    if (num_in != 2)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "Two inputs required ==> (mjd (UT1), longitude (radians)) "
                "[%i inputs found]", num_in);
    }

    // Get matlab inputs.
    double mjd_utc  = mxGetScalar(in[0]);
    double lon_rad  = mxGetScalar(in[1]);

    out[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    double* last_rad = (double*)mxGetPr(out[0]);

    *last_rad = oskar_mjd_to_last_fast_d(mjd_utc, lon_rad);
}
