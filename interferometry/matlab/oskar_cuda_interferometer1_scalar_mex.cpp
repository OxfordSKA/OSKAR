#include <mex.h>
#include <vector>
#include <cmath>
#include <cstring>

#include "interferometry/oskar_cuda_interferometer1_scalar.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Entry function.
void mexFunction(int num_outputs, mxArray ** output, int num_inputs,
        const mxArray ** input)
{
    mexPrintf("num outputs = %i\n", num_outputs);

    if (num_inputs != 1)
        mexErrMsgTxt("One input required.");

    else if (num_outputs > 1)
        mexErrMsgTxt("Too many output arguments.");

    else if (!mxIsStruct(input[0]))
        mexErrMsgTxt("Input must be a structure.");

    mxArray * x = mxGetField(input[0], 0, "Xh");
    mxArray * y = mxGetField(input[0], 0, "Yh");
    double * xh = mxGetPr(x);
    double * yh = mxGetPr(y);
    mexPrintf("%f\n", xh[1]);
    mexPrintf("%f\n", yh[1]);




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
