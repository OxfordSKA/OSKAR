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

#include "interferometry/matlab/oskar_mex_vis_from_matlab_struct.h"

#include "utility/oskar_get_error_string.h"

#include "imaging/oskar_make_image.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_SettingsImage.h"

#include <cstdlib>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** /*out*/, int num_in, const mxArray** in)
{
    if (num_in != 2 || num_out > 1)
    {
        mexErrMsgTxt("Usage: image = oskar_make_image_new(vis, settings)");
    }

    // Load visibilities from MATLAB structure into a oskar_Visibilties structure.
    oskar_Visibilities vis;
    mexPrintf("= loading vis structure... ");
    oskar_mex_vis_from_matlab_struct(&vis, in[0]);
    mexPrintf("done.\n");

    // Construct image settings structure.
    oskar_SettingsImage settings;

    // Setup image object.
    int type = OSKAR_DOUBLE;
    oskar_Image image(type, OSKAR_LOCATION_CPU);

    // Make image.
    int err = oskar_make_image(&image, &vis, &settings);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR",
                "oskar_make_image() failed with code %i: %s.\n",
                err, oskar_get_error_string(err));
    }

    // Convert oskar_Image to MATLAB structure.
    // TODO
}

