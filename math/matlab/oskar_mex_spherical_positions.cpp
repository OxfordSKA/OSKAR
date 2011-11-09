#include <mex.h>
#include "math/oskar_SphericalPositions.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//
// DEPRECATED yes, BUT don't yet delete as some version of this is still useful.
//
// Interface function - can call anything from here...
//
// nlhs = The number of left-hand arguments, or the size of the plhs array.
// plhs = An array of left-hand output arguments.
// nrhs = The number of right-hand arguments, or the size of the prhs array.
// prhs = An array of left-hand output arguments.
void mexFunction(int /* num_outputs */, mxArray ** output,
        int num_inputs, const mxArray ** input)
{
    if (num_inputs != 6)
    {
        mexPrintf("Number of arguments != 6\n");
        mexErrMsgTxt("[long lat] = (centre long, centre_lat, size_long, "
                "size_lat, sep_long, sep_lat) [degrees]");
    }

    const double deg2rad = M_PI / 180.0;

    double centre_long      = mxGetScalar(input[0]) * deg2rad;
    double centre_lat       = mxGetScalar(input[1]) * deg2rad;
    double size_long        = mxGetScalar(input[2]) * deg2rad;
    double size_lat         = mxGetScalar(input[3]) * deg2rad;
    double sep_long         = mxGetScalar(input[4]) * deg2rad;
    double sep_lat          = mxGetScalar(input[5]) * deg2rad;
//    double rho              = mxGetScalar(input[6]);
//    bool force_constant_sep = *mxGetLogicals(input[7]);
//    bool set_centre_after   = *mxGetLogicals(input[8]);
//    bool force_centre_point = *mxGetLogicals(input[9]);
//    bool force_to_edges     = *mxGetLogicals(input[10]);
//    int projection_type     = (int)mxGetScalar(input[11]);

    double rho              = 0.0; // rotation angle in radians?
    bool force_constant_sep = true;
    bool set_centre_after   = false;
    bool force_centre_point = true;
    bool force_to_edges     = true;
    //int projection_type     = SphericalPositions<double>::PROJECTION_SIN;
    int projection_type     = oskar_SphericalPositions<double>::PROJECTION_NONE;

    mexPrintf("Inputs: %d\n", num_inputs);
    mexPrintf("     centre_long = %f (rads)\n", centre_long);
    mexPrintf("     centre_lat  = %f (rads)\n", centre_lat);
    mexPrintf("     size_long   = %f (rads)\n", size_long);
    mexPrintf("     size_lat    = %f (rads)\n", size_lat);
    mexPrintf("     sep_long    = %f (rads)\n", sep_long);
    mexPrintf("     sep_lat     = %f (rads)\n", sep_lat);

    // Setup.
    oskar_SphericalPositions<double> positions(centre_long, centre_lat,
            size_long, size_lat, // Half-widths ?!
            sep_long, sep_lat,
            rho, force_constant_sep, set_centre_after,
            force_centre_point, force_to_edges, projection_type);


    // Dry run to find out how many points.
    unsigned points = positions.generate(0, 0);

    // Allocate memory for output.
    mexPrintf("num points = %i\n", points);
    int rows = 1;
    int cols = points;
    output[0] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxREAL);
    output[1] = mxCreateNumericMatrix(rows, cols, mxDOUBLE_CLASS, mxREAL);
    double * longitudes = mxGetPr(output[0]);
    double * latitudes  = mxGetPr(output[1]);

    // Generate points. todo allocate memory inside this function?
    positions.generate(&longitudes[0], &latitudes[0]);
}
