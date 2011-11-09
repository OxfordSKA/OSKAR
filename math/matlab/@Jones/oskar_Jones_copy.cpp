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

#include "math/oskar_Jones.h"
#include "utility/matlab/oskar_mex_pointer.h"
#include "math/matlab/@Jones/oskar_Jones_utility.h"

// Interface function
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 1 || num_in != 1)
    {
        mexErrMsgTxt("Usage: oskar_Jones = oskar_Jones_copy(jones_pointer)");
    }

    // Extract the oskar_Jones pointer from the mxArray object.
    oskar_Jones* J = covert_mxArray_to_pointer<oskar_Jones>(in[0]);

    // Construct a new oskar_Jones object to copy into as a mxArray.
    // Set up the memory to match the original object.
    int num_sources  = J->num_sources();
    int num_stations = J->num_stations();
    const char* format_string = (J->type() == OSKAR_DOUBLE_COMPLEX_MATRIX ||
            J->type() == OSKAR_SINGLE_COMPLEX_MATRIX) ? "matrix" : "scalar";
    const char* type_string = (J->type() == OSKAR_DOUBLE_COMPLEX_MATRIX ||
            J->type() == OSKAR_DOUBLE_COMPLEX) ? "double" : "single";
    const char* location_string = (J->location() == 0) ? "cpu" : "gpu";
    mxArray* J_class = create_matlab_Jones_class(num_sources, num_stations,
            format_string, type_string, location_string);

    oskar_Jones* J_copy = get_jones_pointer_from_matlab_jones_class(J_class);

    // Copy values into it from the original Jones.
    J->copy_to(J_copy);

    out[0] = J_class;
}
