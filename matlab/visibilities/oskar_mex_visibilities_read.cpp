/*
 * Copyright (c) 2012, The University of Oxford
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

#include "oskar_global.h"

#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_read.h"
#include "interferometry/oskar_visibilities_free.h"

#include "matlab/visibilities/lib/oskar_mex_vis_to_matlab_struct.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_mem_binary_file_read.h"
#include "utility/oskar_binary_tag_index_free.h"

#include <cstdio>
#include <cstdlib>

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out > 1)
        mexErrMsgTxt("Usage: vis = oskar.visibilities.read(filename)");

    // Extract arguments from MATLAB mxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Check the filename exists (... is there better way to do this?)
    mxArray* file_exists;
    mxArray* args[2];
    args[0] = mxCreateString(filename);
    args[1] = mxCreateString("file");
    mexCallMATLAB(1, &file_exists, 2, args, "exist");
    int exists = (int)mxGetScalar(file_exists);
    if (exists == 0)
    {
        mexErrMsgIdAndTxt("OSKAR ERROR", "Specified visibility file (%s)"
                " doesn't exist.\n", filename);
    }

    // Load the OSKAR visibilities structure from the specified file.
    int status = OSKAR_SUCCESS;
    oskar_Visibilities vis;
    oskar_visibilities_read(&vis, filename, &status);
    if (status != OSKAR_SUCCESS)
    {
        mexErrMsgIdAndTxt("OSKAR ERROR",
                "Error reading OSKAR visibilities data file: '%s'.\nERROR: %s.",
                filename, oskar_get_error_string(status));
    }

    /* Read date field from binary file */
    oskar_Mem date(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    oskar_BinaryTagIndex* index = NULL;
    oskar_mem_binary_file_read(&date, filename, &index,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_DATE_TIME_STRING, 0,
            &status);
    oskar_binary_tag_index_free(&index, &status);
    if (status)
    {
        mexErrMsgTxt("ERROR: failed to read date field!\n");
    }

    out[0] = oskar_mex_vis_to_matlab_struct(&vis, &date, filename);

    oskar_visibilities_free(&vis, &status);
    if (status)
    {
        mexErrMsgIdAndTxt("oskar:error", "ERROR: %s\n",
                oskar_get_error_string(status));
    }
}
