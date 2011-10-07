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

#ifndef OSKAR_MEX_JONES_UTILITY_H_
#define OSKAR_MEX_JONES_UTILITY_H_

#include <mex.h>

// Other headers.
#include "math/oskar_Jones.h"
#include "string.h"

// Return the oskar_Jones type for the associated data type and format.
int get_type_id(const char* type, const char* format)
{
    enum { SINGLE, DOUBLE, SCALAR, MATRIX };
    int itype = 0;
    if ( strcmp(type, "single") == 0 )
    {
        itype = SINGLE;
    }
    else if ( strcmp(type, "double") == 0 )
    {
        itype = DOUBLE;
    }
    else
    {
        mexErrMsgTxt("Unrecognised data type. "
                "(accepted values: 'single' or 'double')");
    }

    int iformat = 0;
    if ( strcmp(format, "scalar") == 0 )
    {
        iformat = SCALAR;
    }
    else if ( strcmp(format, "matrix") == 0 )
    {
        iformat = MATRIX;
    }
    else
    {
        mexErrMsgTxt("Unrecognised data format. "
                "(accepted values: 'scalar' or 'matrix')");
    }

    if (itype == SINGLE)
    {
        if (iformat == SCALAR)
            return OSKAR_JONES_FLOAT_SCALAR;
        else
            return OSKAR_JONES_FLOAT_MATRIX;
    }
    else
    {
        if (iformat == SCALAR)
            return OSKAR_JONES_DOUBLE_SCALAR;
        else
            return OSKAR_JONES_DOUBLE_MATRIX;
    }
}

// Return the oskar_Jones type location id for the memory location.
int get_location_id(const char* location)
{
    enum { HOST = 0, DEVICE = 1, UNDEF = -1 };
    if (strcmp(location, "gpu") == 0)
    {
        return DEVICE;
    }
    else if (strcmp(location, "cpu") == 0)
    {
        return HOST;
    }
    else
    {
        mexErrMsgTxt("Unrecognised memory location "
                "(accepted values: 'cpu' or 'gpu')");
    }
    return UNDEF;
}

#endif // OSKAR_MEX_JONES_UTILITY_H_
