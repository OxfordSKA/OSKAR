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
#include <oskar_vector_types.h>
#include <oskar_Mem.h>

#include <string.h>

#define STR_LENGTH 200

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    int err = 0;
    if (num_in < 2 || num_in > 3 || num_out > 1)
    {
        oskar_matlab_usage("[records]", "binary_file", "read_group",
                "<filename>, <group>, [index]", "Reads one or more groups from "
                "the specified OSKAR binary file. If [index] is specified, "
                "only that index is read from the binary, otherwise, all "
                "records in the specified group are read irrespective of "
                "index. Note the <group> can be either string or scalar(int) "
                "type.");
    }

    // Get input arguments.
    char filename[STR_LENGTH];
    mxGetString(in[0], filename, STR_LENGTH);
    union { int id;  char name[STR_LENGTH]; } group;
    bool is_extended = false;
    if (mxIsChar(in[1]))
    {
        is_extended = true;
        mxGetString(in[1], group.name, STR_LENGTH);
    }
    else
    {
        is_extended = false;
        group.id = (int)mxGetScalar(in[1]);
    }

    int read_idx = -1;
    if (num_in == 3)
        read_idx = mxGetScalar(in[2]);

    // Create handle.
    oskar_Binary* h = oskar_binary_create(filename, 'r', &err);
    if (err)
    {
        oskar_matlab_error("oskar_binary_create() failed with code "
                "%i: %s.", err, oskar_get_error_string(err));
    }

    // Find out how many records are in the specified group.
    // with index, if specified.
    int num_records = 0;
    for (int i  = 0; i < h->num_chunks; ++i)
    {
        if (is_extended && h->extended[i])
        {
            if (strcmp(h->name_group[i], group.name) == 0)
            {
                if ((read_idx >= 0 && h->user_index[i] == read_idx) || read_idx < 0)
                    ++num_records;
            }
        }
        else if (!is_extended && !h->extended[i])
        {
            if (h->id_group[i] == group.id)
            {
                if ((read_idx >= 0 && h->user_index[i] == read_idx) || read_idx < 0)
                    ++num_records;
            }
        }
    }

//    if (is_extended && read_idx < 0)
//        mexPrintf("Number of records found with group <%s> = %i\n", group.name,
//                num_records);
//    else if (is_extended && read_idx >= 0)
//        mexPrintf("Number of records found with group <%s> in index <%i> = %i\n",
//                group.name, read_idx, num_records);
//    else if (!is_extended && read_idx < 0)
//        mexPrintf("Number of records found with group <%i> = %i\n", group.id,
//                num_records);
//    else
//        mexPrintf("Number of records found with group <%i> in index <%i> = %i\n",
//                group.id, read_idx, num_records);

    fflush(stdout);

    // Populate MATLAB structure.
    int num_fields = 5;
    const char* fields[5] = { "type", "group", "tag", "index", "data" };
    out[0] = mxCreateStructMatrix(num_records, 1, num_fields, fields);

    for (int k = 0, i  = 0; i < h->num_chunks; ++i)
    {
        // Get the data from the index.
        mxArray* data_ = NULL;
        void* data = NULL;
        mwSize m = 0;
        if (read_idx >= 0 && h->user_index[i] != read_idx)
            continue;

        switch (h->data_type[i])
        {
            case OSKAR_CHAR:
            {
                data = malloc(h->payload_size_bytes[i]);
                break;
            }
            case OSKAR_INT:
            {
                m = h->payload_size_bytes[i] / sizeof(int);
                data_ = mxCreateNumericMatrix(m, 1, mxINT32_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_SINGLE:
            {
                m = h->payload_size_bytes[i] / sizeof(float);
                data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_DOUBLE:
            {
                m = h->payload_size_bytes[i] / sizeof(double);
                data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_SINGLE_COMPLEX:
            {
                m = h->payload_size_bytes[i] / sizeof(float2);
                data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxCOMPLEX);
                data = malloc(h->payload_size_bytes[i]);
                break;
            }
            case OSKAR_DOUBLE_COMPLEX:
            {
                m = h->payload_size_bytes[i] / sizeof(double2);
                data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxCOMPLEX);
                data = malloc(h->payload_size_bytes[i]);
                break;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                m = h->payload_size_bytes[i] / sizeof(float4c);
                mwSize dims[3] = {2, 2, m};
                data_ = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
                data = malloc(h->payload_size_bytes[i]);
                break;
            }
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                m = h->payload_size_bytes[i] / sizeof(double4c);
                mwSize dims[3] = {2, 2, m};
                data_ = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
                data = malloc(h->payload_size_bytes[i]);
                break;
            }
            default:
                oskar_matlab_error("Unknown OSKAR data type");
                break;
        };
        if (h->extended[i])
        {
            oskar_binary_read_ext(h,
                    (unsigned char)h->data_type[i],
                    h->name_group[i],
                    h->name_tag[i],
                    h->user_index[i],
                    (size_t)h->payload_size_bytes[i],
                    data, &err);
        }
        else
        {
            oskar_binary_read(h,
                    (unsigned char)h->data_type[i],
                    (unsigned char)h->id_group[i],
                    (unsigned char)h->id_tag[i],
                    h->user_index[i],
                    (size_t)h->payload_size_bytes[i],
                    data, &err);
        }
        if (err)
        {
            oskar_matlab_error("oskar_binary_file_read() failed with code %i: "
                    "%s", err, oskar_get_error_string(err));
        }

        // If the data is a char array convert to MATLAB string (16bit char format).
        if (h->data_type[i] == OSKAR_CHAR)
        {
            data_ = mxCreateString((char*)data);
            free(data);
        }
        // Convert to MATLAB complex types
        else if (h->data_type[i] == OSKAR_SINGLE_COMPLEX)
        {
            float* re = (float*)mxGetPr(data_);
            float* im = (float*)mxGetPi(data_);
            for (unsigned j = 0; j < (unsigned)m; ++j)
            {
                re[j] = ((float2*)data)[j].x;
                im[j] = ((float2*)data)[j].y;
            }
            free(data);
        }
        else if (h->data_type[i] == OSKAR_DOUBLE_COMPLEX)
        {
            double* re = mxGetPr(data_);
            double* im = mxGetPi(data_);
            for (unsigned j = 0; j < (unsigned)m; ++j)
            {
                re[j] = ((double2*)data)[j].x;
                im[j] = ((double2*)data)[j].y;
            }
            free(data);
        }
        else if (h->data_type[i] == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float* re = (float*)mxGetPr(data_);
            float* im = (float*)mxGetPi(data_);
            for (unsigned j = 0; j < (unsigned)m; ++j)
            {
                re[4*j+0] = ((float4c*)data)[j].a.x;
                im[4*j+0] = ((float4c*)data)[j].a.y;
                re[4*j+1] = ((float4c*)data)[j].c.x;
                im[4*j+1] = ((float4c*)data)[j].c.y;
                re[4*j+2] = ((float4c*)data)[j].b.x;
                im[4*j+2] = ((float4c*)data)[j].b.y;
                re[4*j+3] = ((float4c*)data)[j].d.x;
                im[4*j+3] = ((float4c*)data)[j].d.y;
            }
            free(data);
        }
        else if (h->data_type[i] == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double* re = mxGetPr(data_);
            double* im = mxGetPi(data_);
            for (unsigned j = 0; j < (unsigned)m; ++j)
            {
                re[4*j+0] = ((double4c*)data)[j].a.x;
                im[4*j+0] = ((double4c*)data)[j].a.y;
                re[4*j+1] = ((double4c*)data)[j].c.x;
                im[4*j+1] = ((double4c*)data)[j].c.y;
                re[4*j+2] = ((double4c*)data)[j].b.x;
                im[4*j+2] = ((double4c*)data)[j].b.y;
                re[4*j+3] = ((double4c*)data)[j].d.x;
                im[4*j+3] = ((double4c*)data)[j].d.y;
            }
            free(data);
        }

        if (is_extended && h->extended[i])
        {
            if (strcmp(h->name_group[i], group.name) == 0)
            {
                mxSetField(out[0], k, fields[0],
                        mxCreateString(oskar_mem_data_type_string(h->data_type[i])));
                mxSetField(out[0], k, fields[1], mxCreateString(h->name_group[i]));
                mxSetField(out[0], k, fields[2], mxCreateString(h->name_tag[i]));
                mxSetField(out[0], k, fields[3], mxCreateDoubleScalar((double)h->user_index[i]));
                mxSetField(out[0], k, fields[4], data_);
                ++k;
            }
        }
        else if (!is_extended && !h->extended[i])
        {
            if (h->id_group[i] == group.id)
            {
//                mexPrintf("RECORD: i = %i/%i, k = %i/%i\n", i, index->num_tags, k, num_records);
                mxSetField(out[0], k, fields[0],
                        mxCreateString(oskar_mem_data_type_string(h->data_type[i])));
                mxSetField(out[0], k, fields[1], mxCreateDoubleScalar((double)h->id_group[i]));
                mxSetField(out[0], k, fields[2], mxCreateDoubleScalar((double)h->id_tag[i]));
                mxSetField(out[0], k, fields[3], mxCreateDoubleScalar((double)h->user_index[i]));
                mxSetField(out[0], k, fields[4], data_);
                ++k;
            }
        }
    }

    // Clean up
    oskar_binary_free(h);
}
