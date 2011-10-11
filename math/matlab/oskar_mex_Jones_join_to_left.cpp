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
#include "math/matlab/oskar_mex_pointer.h"
#include "math/oskar_jones_join.h"

// Interface function
void mexFunction(int num_out,  mxArray** out, int num_in, const mxArray** in)
{
    // Check arguments.
    if (num_out != 0 || num_in != 2)
    {
        // other = other * this
        mexErrMsgTxt("Usage: oskar_Jones_join_to_left(this.pointer, other.pointer)");
    }

    // Extract the oskar_Jones pointers from the mxArray object.
    oskar_Jones* Jthis  = covert_mxArray_to_pointer<oskar_Jones>(in[0]);
    oskar_Jones* Jother = covert_mxArray_to_pointer<oskar_Jones>(in[1]);

    if (Jthis->n_sources() != Jother->n_sources())
    {
        mexErrMsgTxt("Unable to join two matrices with different source dimensions!");
    }

    if (Jthis->n_stations() != Jother->n_stations())
    {
        mexErrMsgTxt("Unable to join two matrices with different station dimensions!");
    }

    if (Jthis->type() != Jother->type())
    {
        mexErrMsgTxt("Unable to join two matrices of different type");
    }

    int err = Jthis->join_to_left(Jother);

    if (err != 0)
    {
        mexPrintf("oskar_jones_join returned error code %i\n", err);
        mexErrMsgTxt("Failed to complete join");
    }
}
