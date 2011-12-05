/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <mex.h>
#include <cstring>

#include "utility/oskar_vector_types.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "utility/oskar_Mem.h"
#include "utility/matlab/oskar_get_type_string.h"

// MATLAB entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check number of input arguments.
    if (num_out != 1 || num_in != 4)
    {
        mexErrMsgTxt("Usage: pointer = oskar_mem_set_values(pointer, type, "
                "location, values)");
    }

    // Parse input arguments.
    oskar_Mem* mem     = covert_mxArray_to_pointer<oskar_Mem>(in[0]);
    int type           = (int)mxGetScalar(in[1]);
    int location       = (int)mxGetScalar(in[2]);
    int num_elements   = mem->num_elements();

    // Properties of the values array.
    mwSize num_dims    = mxGetNumberOfDimensions(in[3]);
    const mwSize* dims = mxGetDimensions(in[3]);
    mxClassID class_id = mxGetClassID(in[3]);
    bool complex       = mxIsComplex(in[3]);

    // Only support 3 or 2 dimensions for matrix data and 1 dimension for scalar
    // data.
    if (num_dims > 3)
    {
        mexErrMsgTxt("Specified values array has unsupported dimensions");
    }


    // Work out the OSKAR type of the values array.
    int values_type = 0;
    switch (class_id)
    {
        case mxSINGLE_CLASS:
            values_type = OSKAR_SINGLE;
            break;
        case mxDOUBLE_CLASS:
            values_type = OSKAR_DOUBLE;
            break;
        case mxINT32_CLASS:
            values_type = OSKAR_INT;
            break;
        default:
            mexErrMsgTxt("Incompatible data type for specified values array");
            break;
    };

    mexPrintf("values type = %i %s\n", values_type, oskar_get_type_string(values_type));

    if (complex && values_type != OSKAR_INT)
    {
        values_type |= OSKAR_COMPLEX;
    }
    mexPrintf("values type = %i %s\n", values_type, oskar_get_type_string(values_type));

    // Work out if this is matrix data.
    // Matrix data is (2,2,n) where for the case of n = 1 the dimension will
    // be collapsed.
    bool matrix = false;
    if (num_dims >=2 && num_dims <= 3)
    {
        if (dims[0] == 2 && dims[1] == 2)
            matrix = true;
        else
            mexErrMsgTxt("Values array has invalid dimensions");
    }
    if (matrix)
    {
        values_type |= OSKAR_MATRIX;
    }
    mexPrintf("values type = %i %s\n", values_type, oskar_get_type_string(values_type));









}

