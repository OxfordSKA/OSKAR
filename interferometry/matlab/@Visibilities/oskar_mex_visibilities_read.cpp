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

#include "oskar_global.h"

#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"
#include "interferometry/matlab/oskar_mex_vis_to_matlab_struct.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_mem_binary_file_read.h"
#include "utility/oskar_binary_tag_index_free.h"
#include <cstdio>
#include <cstdlib>

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out > 1)
        mexErrMsgTxt("Usage: vis = oskar_visibilities_read(filename)");

    // Extract arguments from MATLAB mxArray objects.
    const char* filename = mxArrayToString(in[0]);

    // Load the OSKAR visibilities structure from the specified file.
    int status = OSKAR_SUCCESS;
    oskar_Visibilities vis;
    status = oskar_Visibilities::read(&vis, filename);

    if (status != OSKAR_SUCCESS)
    {
        mexErrMsgIdAndTxt("OSKAR:error",
                "Error reading OSKAR visibilities data file: '%s'.\nERROR: %s.",
                filename, oskar_get_error_string(status));
    }

    /* Read date field from binary file */
    oskar_Mem date(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    oskar_BinaryTagIndex* index = NULL;
    int err = oskar_mem_binary_file_read(&date, filename, &index,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_DATE_TIME_STRING, 0);
    oskar_binary_tag_index_free(&index);
    if (err)
    {
        mexErrMsgTxt("ERROR: failed to read date field!\n");
    }

    out[0] = oskar_mex_vis_to_matlab_struct(&vis, &date);
}
