/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_vis.h>

#include <oskar_get_error_string.h>
#include <oskar_binary.h>
#include <oskar_binary_read_mem.h>

#include "matlab/vis/lib/oskar_mex_vis_to_matlab_struct.h"
#include "matlab/common/oskar_matlab_common.h"

#include <cstdio>
#include <cstdlib>

void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out > 1) {
        oskar_matlab_usage("[vis]", "vis", "read", "<filename>",
                "Reads an OSKAR visibilities binary data file into "
                "a MATLAB structure");
    }

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
        oskar_matlab_error("Specified visibility file (%s) doesn't exist.",
                filename);
    }

    // Load the OSKAR visibilities structure from the specified file.
    int status = OSKAR_SUCCESS;
    oskar_Binary* index = oskar_binary_create(filename, 'r', &status);
    oskar_Vis* vis = oskar_vis_read(index, &status);
    if (status != OSKAR_SUCCESS)
    {
        oskar_matlab_error("Failed reading OSKAR visibility data file: '%s' (%s)",
                filename, oskar_get_error_string(status));
    }

    /* Read date field from binary file */
    oskar_Mem *date = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &status);
    oskar_binary_read_mem(index, date, OSKAR_TAG_GROUP_METADATA,
            OSKAR_TAG_METADATA_DATE_TIME_STRING, 0, &status);
    oskar_binary_free(index);
    if (status)
    {
        oskar_matlab_error("Failed to read date field!");
    }

    out[0] = oskar_mex_vis_to_matlab_struct(vis, date, filename);

    oskar_vis_free(vis, &status);
    if (status)
    {
        oskar_matlab_error("Failed reading OSKAR visibility file: %s",
                oskar_get_error_string(status));
    }
    oskar_mem_free(date, &status);
}
