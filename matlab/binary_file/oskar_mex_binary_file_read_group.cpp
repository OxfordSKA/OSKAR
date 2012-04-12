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
#include "utility/oskar_Mem.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_data_type_string.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_vector_types.h"

#include <string.h>

#define STR_LENGTH 200

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 2 || num_out > 1)
    {
        mexErrMsgTxt("Usage: records = read_group(filename, group)\n");
    }

    int err = OSKAR_SUCCESS;


    // Get input arguments.
    char filename[STR_LENGTH];
    mxGetString(in[0], filename, STR_LENGTH);
    union { int id;  char name[STR_LENGTH]; } group;
    if (mxIsChar(in[1]))
    {
        mxGetString(in[1], group.name, STR_LENGTH);
    }
    else
    {
        group.id = (int)mxGetScalar(in[1]);
    }



    // Open the binary file for reading.
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "Unable to open file (%s)\n.", filename);
    }


    // Create binary tag index (the record header?)
    oskar_BinaryTagIndex* index = NULL;
    err = oskar_binary_tag_index_create(&index, file);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_tag_index_create() "
                "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }


    // Find out how many records are in the specified group.
    int num_records = 0;
    // TODO ...




    // Populate MATLAB structure.
    int num_fields = 5;
    const char* fields[5] = {
            "type",
            "group",
            "tag",
            "index",
            "data"
    };
    // TODO ...

    // Clean up
    oskar_binary_tag_index_free(&index);
    fclose(file);
}


