#include <mex.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "interferometry/oskar_interferometer1_scalar.h"
#include "interferometry/oskar_horizon_plane_to_itrs.h"

#include "interferometry/oskar_TelescopeModel.h"
#include "beamforming/oskar_StationModel.h"
#include "sky/oskar_SkyModel.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


void getTelescopeModel(const mxArray * matlab_struct, oskar_TelescopeModel * model);
void getStationModels(const mxArray * matlab_struct, oskar_StationModel * model);

// Entry function.
void mexFunction(int /*num_outputs*/, mxArray ** /*output*/, int num_inputs,
        const mxArray ** input)
{
    // ==== Parse Inputs.
    if (num_inputs != 2)
        mexErrMsgTxt("Two inputs required.");

    // ==== Construct telescope model structure from MATLAB structure.
    oskar_TelescopeModel telescope;
    getTelescopeModel(input[0], &telescope);






//    for (unsigned i = 0; i < telescope.num_antennas; ++i)
//        mexPrintf("%i %f %f %f\n", i, telescope.antenna_x[i], telescope.antenna_y[i],
//                telescope.antenna_z[i]);

    // ==== Construct array of station models from MATLAB structure.
    if (mxGetN(input[1]) != telescope.num_antennas)
        mexErrMsgTxt("Dimension mismatch between telescope model and station models");
    size_t mem_size = telescope.num_antennas * sizeof(oskar_StationModel);
    oskar_StationModel * stations = (oskar_StationModel*) malloc(mem_size);
    getStationModels(input[1], stations);






    // TODO: call oskar_cudad_interferometer1_scalar()






    // TODO: Free memory for local CPU arrays (telescope model).




//    SkyModel sky;
//
//    const double ra0_rads = 0;
//    const double dec0_rads = M_PI / 2.0;
//
//    const double start_date_utc = 1.0;
//    const unsigned nsdt = 0;
//    const double sdt = 0.0;
//
//    const double lamba_bandwidth = 1.0;
//
//    const unsigned num_baselines = num_stations * (num_stations - 1) / 2;
//    double * vis = (double*) malloc(num_baselines * sizeof(double));
//
//    oskar_cudad_interferometer1_scalar(telescope, stations, sky,
//            ra0_rads, dec0_rads, start_date_utc, nsdt, sdt, lamba_bandwidth, vis);
//
//    mexPrintf("%f %f\n", vis[0], vis[1]);

//    // get input arguments.
//    int nfields = mxGetNumberOfFields(input[0]);
//    mwSize NStructElems = mxGetNumberOfElements(input[0]);
//
//    // allocate memory  for storing classIDflags.
//    mxClassID * classIDflags = (mxClassID*) mxCalloc(nfields, sizeof(mxClassID));
//
//    // check empty field, proper data type, and data type consistency
//    // and get classID for each field.
//    for(int ifield=0; ifield < nfields; ++ifield)
//    {
//        for(mwIndex jstruct = 0; jstruct < NStructElems; ++jstruct)
//        {
//            mxArray * tmp = mxGetFieldByNumber(input[0], jstruct, ifield);
//
//            if (tmp == NULL)
//            {
//                mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
//                mexErrMsgTxt("Above field is empty!");
//            }
//            if (jstruct == 0)
//            {
//                if( (!mxIsChar(tmp) && !mxIsNumeric(tmp)) || mxIsSparse(tmp))
//                {
//                    mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
//                    mexErrMsgTxt("Above field must have either string or numeric non-sparse data.");
//                }
//                classIDflags[ifield]=mxGetClassID(tmp);
//            }
//            else
//            {
//                if (mxGetClassID(tmp) != classIDflags[ifield])
//                {
//                    mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
//                    mexErrMsgTxt("Inconsistent data type in above field!");
//                }
//                else if(!mxIsChar(tmp) && ((mxIsComplex(tmp) || mxGetNumberOfElements(tmp)!=1)))
//                {
//                    mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
//                    mexErrMsgTxt("Numeric data in above field must be scalar and noncomplex!");
//                }
//            }
//        }
//    }
//
//    // allocate memory  for storing pointers.
//    const char ** fnames = (const char**)mxCalloc(nfields, sizeof(*fnames));
//
//    // get field name pointers
//    for (int ifield=0; ifield< nfields; ++ifield)
//    {
//        fnames[ifield] = mxGetFieldNameByNumber(input[0], ifield);
//        mexPrintf("%s\n", fnames[ifield]);
//    }
//
//
//    // create a 1x1 struct matrix for output.
//    output[0] = mxCreateStructMatrix(1, 1, nfields, fnames);
//
//    mxFree((void *)fnames);
//
//
//    mwSize ndim = mxGetNumberOfDimensions(output[0]);
//    const mwSize * dims = mxGetDimensions(input[0]);
//
//    char * pdata = NULL;
//    mxArray * fout;
//    for (int ifield = 0; ifield < nfields; ++ifield)
//    {
//        // create cell/numeric array.
//        if (classIDflags[ifield] == mxCHAR_CLASS)
//        {
//            fout = mxCreateCellArray(ndim, dims);
//        }
//        else {
//            fout = mxCreateNumericArray(ndim, dims, classIDflags[ifield], mxREAL);
//            pdata = (char*)mxGetData(fout);
//        }
//
//        // copy data from input structure array.
//        for (mwIndex jstruct=0; jstruct < NStructElems; ++jstruct)
//        {
//            mxArray * tmp = mxGetFieldByNumber(input[0], jstruct, ifield);
//
//            if( mxIsChar(tmp))
//            {
//                mxSetCell(fout, jstruct, mxDuplicateArray(tmp));
//            }
//            else {
//                mwSize     sizebuf;
//                sizebuf = mxGetElementSize(tmp);
//                memcpy(pdata, mxGetData(tmp), sizebuf);
//                pdata += sizebuf;
//            }
//        }
//        /* set each field in output structure */
//        mxSetFieldByNumber(output[0], 0, ifield, fout);
//    }
//    mxFree(classIDflags);
}



void getTelescopeModel(const mxArray * matlab_struct, oskar_TelescopeModel * model)
{
    // TODO parse structure properly - i.e. check fields etc.
    if (!mxIsStruct(matlab_struct))
        mexErrMsgTxt("Input must be a matlab struct");

    const unsigned num_antennas = (unsigned)mxGetM(mxGetField(matlab_struct, 0, "Xh"));
    const double * xh = mxGetPr(mxGetField(matlab_struct, 0, "Xh"));
    const double * yh = mxGetPr(mxGetField(matlab_struct, 0, "Yh"));
    const double lat  = mxGetScalar(mxGetField(matlab_struct, 0, "latitude"));

    // Allocate memory for telescope model.
    model->num_antennas = num_antennas;
    size_t mem_size = num_antennas * sizeof(double);
    model->antenna_x = (double*)malloc(mem_size);
    model->antenna_y = (double*)malloc(mem_size);
    model->antenna_z = (double*)malloc(mem_size);

    // Convert horizon x, y coordinates to ITRS (local equatorial system)
    oskar_horizon_plane_to_itrs(num_antennas, xh, yh, lat, model->antenna_x,
            model->antenna_y, model->antenna_z);
}

void getStationModels(const mxArray * matlab_struct, oskar_StationModel * model)
{
    if (!mxIsStruct(matlab_struct))
        mexErrMsgTxt("Input must be a matlab struct");

    const unsigned num_stations = mxGetN(matlab_struct);
    for (unsigned i = 0; i < num_stations; ++i)
    {
        model[i].num_antennas = mxGetM(mxGetField(matlab_struct, i, "X"));
        model[i].antenna_x = mxGetPr(mxGetField(matlab_struct, i, "X"));
        model[i].antenna_y = mxGetPr(mxGetField(matlab_struct, i, "Y"));
    }
}

