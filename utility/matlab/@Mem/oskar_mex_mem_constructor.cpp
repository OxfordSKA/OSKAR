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
#include <matrix.h>
#include <mat.h>

#include "utility/oskar_Mem.h"
#include "utility/matlab/oskar_mex_pointer.h"

// MATLAB entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_out > 1 || num_in != 4)
    {
        mexErrMsgTxt("Usage: mem_pointer = oskar_mem_constructor(type, "
                "location, num_elements, owner)");
    }

    // Get arguments.
    int type         = (int)mxGetScalar(in[0]);
    int location     = (int)mxGetScalar(in[1]);
    int num_elements = (int)mxGetScalar(in[2]);
    int owner        = (int)mxGetScalar(in[3]);

    // Check the type is valid.
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE && type != OSKAR_INT &&
            type != OSKAR_SINGLE_COMPLEX && type != OSKAR_DOUBLE_COMPLEX &&
            type != OSKAR_SINGLE_COMPLEX_MATRIX &&
            type != OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        mexErrMsgIdAndTxt("OSKAR:error", "Unknown oskar_Mem type ID %i",
                type);
    }

    // Check Location is valid.
    if (location != OSKAR_LOCATION_CPU && location != OSKAR_LOCATION_GPU)
    {
        mexErrMsgIdAndTxt("OSKAR:error", "Unknown oskar_Mem location ID %i",
                location);
    }

    // Create the oskar_Mem pointer.
    oskar_Mem* m = NULL;
    try {
        m = new oskar_Mem(type, location, num_elements, owner);
    }
    catch (const char* error)
    {
        mexErrMsgIdAndTxt("OSKAR:error", "%s", error);
    }

    // Convert the mem pointer to an mxArray uint64 pointer wrapper.
    out[0] = convert_pointer_to_mxArray(m);
}

