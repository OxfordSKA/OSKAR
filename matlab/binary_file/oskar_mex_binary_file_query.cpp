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

#include "matlab/common/oskar_matlab_common.h"

#include <mex.h>

#include <private_binary.h>
#include <oskar_binary.h>
#include <oskar_get_error_string.h>
#include <oskar_Mem.h>

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    int err = 0;
    if (num_in != 1 || num_out > 2)
    {
        oskar_matlab_usage("[index, headers]", "binary_file", "query",
                "<filename>", "Queries the contents of an OSKAR binary file "
                "returning a cell array of record indices and an array of "
                "record header structures.");
    }

    // Get input args
    char filename[100];
    mxGetString(in[0], filename, 100);

    // Run 'format long' command
    mxArray* arg = mxCreateString("long");
    mexCallMATLAB(0, NULL, 1, &arg, "format");

    oskar_Binary* h = oskar_binary_create(filename, 'r', &err);
    if (err) oskar_matlab_error("oskar_binary_create() failed with "
            "code %i: %s",err, oskar_get_error_string(err));

    mexPrintf("= Binary file header loaded (%i tags found).\n", h->num_chunks);
    int num_fields = 5;
    const char* fields[5] = { "type", "group", "tag", "index", "data_size"};

    int num_records = h->num_chunks;
    out[0] = mxCreateCellMatrix(num_records + 1, 6);
    out[1] = mxCreateStructMatrix(num_records, 1, num_fields, fields);

    mxSetCell(out[0], 0 * (num_records + 1), mxCreateString("[ID]"));
    mxSetCell(out[0], 1 * (num_records + 1), mxCreateString("[TYPE]"));
    mxSetCell(out[0], 2 * (num_records + 1), mxCreateString("[GROUP]"));
    mxSetCell(out[0], 3 * (num_records + 1), mxCreateString("[TAG]"));
    mxSetCell(out[0], 4 * (num_records + 1), mxCreateString("[INDEX]"));
    mxSetCell(out[0], 5 * (num_records + 1), mxCreateString("[BYTES]"));

    for (int i = 0; i < num_records; ++i)
    {
        mxSetCell(out[0], 0 * (num_records + 1) + i + 1, mxCreateDoubleScalar((double)i+1));
        const char* data_name = oskar_mem_data_type_string(h->data_type[i]);
        mxSetField(out[1], i, fields[0], mxCreateString(data_name));
        mxSetCell(out[0], 1 * (num_records + 1) + i + 1, mxCreateString(data_name));
        if (h->extended[i])
        {
            mxSetField(out[1], i, fields[1],
                    mxCreateString(h->name_group[i]));
            mxSetCell(out[0], 2 * (num_records + 1) + i + 1, mxCreateString(h->name_group[i]));
            mxSetField(out[1], i, fields[2],
                    mxCreateString(h->name_tag[i]));
            mxSetCell(out[0], 3 * (num_records + 1) + i + 1, mxCreateString(h->name_tag[i]));
        }
        else
        {
            mxSetField(out[1], i, fields[1],
                    mxCreateDoubleScalar((double)h->id_group[i]));
            mxSetCell(out[0], 2 * (num_records + 1) + i + 1,
                    mxCreateDoubleScalar((double)h->id_group[i]));
            mxSetField(out[1], i, fields[2],
                    mxCreateDoubleScalar((double)h->id_tag[i]));
            mxSetCell(out[0], 3 * (num_records + 1) + i + 1,
                    mxCreateDoubleScalar((double)h->id_tag[i]));
        }
        mxSetField(out[1], i, fields[3],
                mxCreateDoubleScalar((double)h->user_index[i]));
        mxSetCell(out[0], 4 * (num_records + 1) + i + 1,
                mxCreateDoubleScalar((double)h->user_index[i]));
        mxSetField(out[1], i, fields[4],
                mxCreateDoubleScalar((double)h->payload_size_bytes[i]));
        mxSetCell(out[0], 5 * (num_records + 1) + i + 1,
                mxCreateDoubleScalar((double)h->payload_size_bytes[i]));
    }
    oskar_binary_free(h);
}
