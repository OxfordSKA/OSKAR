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

// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in < 1 || num_in > 4 || num_out > 1)
    {
        mexErrMsgTxt("Usage: record = oskar_binary_file_read_record(filename,"
                " [group id = 0], [tag id = 0], [index id = 0])\n");
    }

    /* Get input args */
    char filename[100];
    int group_id = 0, tag_id = 0, index_id = 0;
    mxGetString(in[0], filename, 100);
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "Unable to open file (%s)\n.", filename);
    }
    if (num_in == 2) group_id = (int)mxGetScalar(in[1]);
    if (num_in == 3)
    {
        group_id = (int)mxGetScalar(in[1]);
        tag_id = (int)mxGetScalar(in[2]);
    }
    if (num_in == 4)
    {
        group_id = (int)mxGetScalar(in[1]);
        tag_id = (int)mxGetScalar(in[2]);
        index_id = (int)mxGetScalar(in[3]);
    }

    mexPrintf("= Reading record %i.%i.%i\n", group_id, tag_id, index_id);

    oskar_BinaryTagIndex* index = NULL;
    int err = oskar_binary_tag_index_create(&index, file);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_tag_index_create() "
                "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }

    int idx = -1;
    for (int i = 0; i < index->num_tags; ++i)
    {
        if (index->id_group[i] == group_id && index->id_tag[i] == tag_id &&
                index->user_index[i] == index_id)
        {
            idx = i;
            break;
        }
    }

    if (idx == -1)
    {
        mexErrMsgTxt("ERROR: Specified group, tag and index not found in binary file\n");
    }

    if (index->extended[idx])
    {
        mexErrMsgTxt("ERROR: extended tags currently not handled by this function\n");
    }

    const char* fields[12] = {
            "extended",
            "data_type",
            "data_type_name",
            "group_id",
            "group_name",
            "tag_id",
            "tag_name",
            "user_index",
            "data_offset",
            "data_size",
            "block_size",
            "data"
    };
    out[0] = mxCreateStructMatrix(1, 1, 12, fields);
    mxSetField(out[0], 0, fields[0],
            mxCreateLogicalScalar(index->extended[idx] ? true : false));
    mxSetField(out[0], 0, fields[1],
            mxCreateDoubleScalar((double)index->data_type[idx]));
    mxSetField(out[0], 0, fields[2],
            mxCreateString(oskar_get_data_type_string(index->data_type[idx])));
    mxSetField(out[0], 0, fields[3],
            mxCreateDoubleScalar((double)index->id_group[idx]));
    mxSetField(out[0], 0, fields[4],
            mxCreateString(index->name_group[idx]));
    mxSetField(out[0], 0, fields[5],
            mxCreateDoubleScalar((double)index->id_tag[idx]));
    mxSetField(out[0], 0, fields[6],
            mxCreateString(index->name_tag[idx]));
    mxSetField(out[0], 0, fields[7],
            mxCreateDoubleScalar((double)index->user_index[idx]));
    mxSetField(out[0], 0, fields[8],
            mxCreateDoubleScalar((double)index->data_offset_bytes[idx]));
    mxSetField(out[0], 0, fields[9],
            mxCreateDoubleScalar((double)index->data_size_bytes[idx]));
    mxSetField(out[0], 0, fields[10],
            mxCreateDoubleScalar((double)index->block_size_bytes[idx]));


    mxArray* data_ = NULL;
    void* data = NULL;
    mwSize m = 0;
    switch (index->data_type[idx])
    {
        case OSKAR_CHAR:
        {
            data = malloc(index->data_size_bytes[idx]);
            break;
        }
        case OSKAR_INT:
        {
            m = index->data_size_bytes[idx] / sizeof(int);
            data_ = mxCreateNumericMatrix(m, 1, mxINT32_CLASS, mxREAL);
            data = mxGetData(data_);
            break;
        }
        case OSKAR_SINGLE:
        {
            m = index->data_size_bytes[idx] / sizeof(float);
            data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxREAL);
            data = mxGetData(data_);
            break;
        }
        case OSKAR_DOUBLE:
        {
            m = index->data_size_bytes[idx] / sizeof(double);
            data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
            data = mxGetData(data_);
            break;
        }
        case OSKAR_SINGLE_COMPLEX:
        {
            m = index->data_size_bytes[idx] / sizeof(float2);
            data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxCOMPLEX);
            data = malloc(index->data_size_bytes[idx]);
            break;
        }
        case OSKAR_DOUBLE_COMPLEX:
        {
            m = index->data_size_bytes[idx] / sizeof(double2);
            data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxCOMPLEX);
            data = malloc(index->data_size_bytes[idx]);
            break;
        }
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            m = index->data_size_bytes[idx] / sizeof(float4c);
            mwSize dims[3] = {2, 2, m};
            data_ = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
            data = malloc(index->data_size_bytes[idx]);
            break;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        {
            m = index->data_size_bytes[idx] / sizeof(double4c);
            mwSize dims[3] = {2, 2, m};
            data_ = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
            data = malloc(index->data_size_bytes[idx]);
            break;
        }
        default:
            mexErrMsgTxt("Unknown OSKAR data type");
            break;
    };

    err = oskar_binary_stream_read(file, &index, index->data_type[idx], group_id,
            tag_id, index_id, index->data_size_bytes[idx], data);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_file_read() "
                "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }

    /* If the data is a char array convert to MATLAB string (16bit char format). */
    if (index->data_type[idx] == OSKAR_CHAR)
    {
        data_ = mxCreateString((char*)data);
        free(data);
    }
    else if (index->data_type[idx] == OSKAR_SINGLE_COMPLEX)
    {
        float* re = (float*)mxGetPr(data_);
        float* im = (float*)mxGetPi(data_);
        for (unsigned i = 0; i < m; ++i)
        {
            re[i] = ((float2*)data)[i].x;
            im[i] = ((float2*)data)[i].y;
        }
        free(data);
    }
    else if (index->data_type[idx] == OSKAR_DOUBLE_COMPLEX)
    {
        double* re = mxGetPr(data_);
        double* im = mxGetPi(data_);
        for (unsigned i = 0; i < m; ++i)
        {
            re[i] = ((double2*)data)[i].x;
            im[i] = ((double2*)data)[i].y;
        }
        free(data);
    }
    else if (index->data_type[idx] == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        float* re = (float*)mxGetPr(data_);
        float* im = (float*)mxGetPi(data_);
        for (unsigned i = 0; i < m; ++i)
        {
            re[4*i+0] = ((float4c*)data)[i].a.x;
            im[4*i+0] = ((float4c*)data)[i].a.y;
            re[4*i+1] = ((float4c*)data)[i].b.x;
            im[4*i+1] = ((float4c*)data)[i].b.y;
            re[4*i+2] = ((float4c*)data)[i].c.x;
            im[4*i+2] = ((float4c*)data)[i].c.y;
            re[4*i+3] = ((float4c*)data)[i].d.x;
            im[4*i+3] = ((float4c*)data)[i].d.y;
        }
        free(data);
    }
    else if (index->data_type[idx] == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        double* re = mxGetPr(data_);
        double* im = mxGetPi(data_);
        for (unsigned i = 0; i < m; ++i)
        {
            re[4*i+0] = ((double4c*)data)[i].a.x;
            im[4*i+0] = ((double4c*)data)[i].a.y;
            re[4*i+1] = ((double4c*)data)[i].b.x;
            im[4*i+1] = ((double4c*)data)[i].b.y;
            re[4*i+2] = ((double4c*)data)[i].c.x;
            im[4*i+2] = ((double4c*)data)[i].c.y;
            re[4*i+3] = ((double4c*)data)[i].d.x;
            im[4*i+3] = ((double4c*)data)[i].d.y;
        }
        free(data);
    }
    mxSetField(out[0], 0, fields[11], data_);

    fclose(file);
    err = oskar_binary_tag_index_free(&index);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_tag_index_free() "
                "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }
}


