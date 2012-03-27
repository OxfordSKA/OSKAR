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
    if (num_in != 1 || num_out > 1)
    {
        mexErrMsgTxt("Usage: data = oskar_binary_file_read(filename)\n");
    }

    int err = OSKAR_SUCCESS;

    /* Get input args */
    char filename[100];
    mxGetString(in[0], filename, 100);
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "Unable to open file (%s)\n.", filename);
    }

    oskar_BinaryTagIndex* index = NULL;
    err = oskar_binary_tag_index_create(&index, file);
    if (err)
    {
        mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_tag_index_create() "
                "failed with code %i: %s.\n", err, oskar_get_error_string(err));
    }

    mexPrintf("= Binary file header loaded (%i tags found).\n", index->num_tags);

    int num_fields = 5;
    const char* fields[5] = {
            "type",
            "group",
            "tag",
            "index",
            "data"
    };
    out[0] = mxCreateStructMatrix(index->num_tags, 1, num_fields, fields);
    for (int i = 0; i < index->num_tags; ++i)
    {
        mxSetField(out[0], i, fields[0],
                mxCreateString(oskar_get_data_type_string(index->data_type[i])));
        if (index->extended[i])
        {
            mxSetField(out[0], i, fields[1],
                    mxCreateString(index->name_group[i]));
            mxSetField(out[0], i, fields[2],
                    mxCreateString(index->name_tag[i]));
        }
        else
        {
            mxSetField(out[0], i, fields[1],
                    mxCreateDoubleScalar((double)index->id_group[i]));
            mxSetField(out[0], i, fields[2],
                    mxCreateDoubleScalar((double)index->id_tag[i]));
        }
        mxSetField(out[0], i, fields[3],
                mxCreateDoubleScalar((double)index->user_index[i]));
        mxArray* data_ = NULL;
        void* data = NULL;
        mwSize m = 0;
        switch (index->data_type[i])
        {
            case OSKAR_CHAR:
            {
                data = malloc(index->data_size_bytes[i]);
                break;
            }
            case OSKAR_INT:
            {
                m = index->data_size_bytes[i] / sizeof(int);
                data_ = mxCreateNumericMatrix(m, 1, mxINT32_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_SINGLE:
            {
                m = index->data_size_bytes[i] / sizeof(float);
                data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_DOUBLE:
            {
                m = index->data_size_bytes[i] / sizeof(double);
                data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxREAL);
                data = mxGetData(data_);
                break;
            }
            case OSKAR_SINGLE_COMPLEX:
            {
                m = index->data_size_bytes[i] / sizeof(float2);
                data_ = mxCreateNumericMatrix(m, 1, mxSINGLE_CLASS, mxCOMPLEX);
                data = malloc(index->data_size_bytes[i]);
                break;
            }
            case OSKAR_DOUBLE_COMPLEX:
            {
                m = index->data_size_bytes[i] / sizeof(double2);
                data_ = mxCreateNumericMatrix(m, 1, mxDOUBLE_CLASS, mxCOMPLEX);
                data = malloc(index->data_size_bytes[i]);
                break;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                m = index->data_size_bytes[i] / sizeof(float4c);
                mwSize dims[3] = {2, 2, m};
                data_ = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxCOMPLEX);
                data = malloc(index->data_size_bytes[i]);
                break;
            }
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                m = index->data_size_bytes[i] / sizeof(double4c);
                mwSize dims[3] = {2, 2, m};
                data_ = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX);
                data = malloc(index->data_size_bytes[i]);
                break;
            }
            default:
                mexErrMsgTxt("Unknown OSKAR data type");
                break;
        };
        if (index->extended[i])
        {
            err = oskar_binary_stream_read_ext(file, &index,
                    (unsigned char)index->data_type[i],
                    index->name_group[i],
                    index->name_tag[i],
                    index->user_index[i],
                    (size_t)index->data_size_bytes[i],
                    data);
        }
        else
        {
            err = oskar_binary_stream_read(file, &index,
                    (unsigned char)index->data_type[i],
                    (unsigned char)index->id_group[i],
                    (unsigned char)index->id_tag[i],
                    index->user_index[i],
                    (size_t)index->data_size_bytes[i],
                    data);
        }
        if (err)
        {
            mexErrMsgIdAndTxt("OSKAR:ERROR", "ERROR: oskar_binary_file_read() "
                    "failed with code %i: %s.\n", err, oskar_get_error_string(err));
        }
        /* If the data is a char array convert to MATLAB string (16bit char format). */
        if (index->data_type[i] == OSKAR_CHAR)
        {
            data_ = mxCreateString((char*)data);
            free(data);
        }
        else if (index->data_type[i] == OSKAR_SINGLE_COMPLEX)
        {
            float* re = (float*)mxGetPr(data_);
            float* im = (float*)mxGetPi(data_);
            for (unsigned j = 0; j < m; ++j)
            {
                re[j] = ((float2*)data)[j].x;
                im[j] = ((float2*)data)[j].y;
            }
            free(data);
        }
        else if (index->data_type[i] == OSKAR_DOUBLE_COMPLEX)
        {
            double* re = mxGetPr(data_);
            double* im = mxGetPi(data_);
            for (unsigned j = 0; j < m; ++j)
            {
                re[j] = ((double2*)data)[j].x;
                im[j] = ((double2*)data)[j].y;
            }
            free(data);
        }
        else if (index->data_type[i] == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float* re = (float*)mxGetPr(data_);
            float* im = (float*)mxGetPi(data_);
            for (unsigned j = 0; j < m; ++j)
            {
                re[4*j+0] = ((float4c*)data)[j].a.x;
                im[4*j+0] = ((float4c*)data)[j].a.y;
                re[4*j+1] = ((float4c*)data)[j].b.x;
                im[4*j+1] = ((float4c*)data)[j].b.y;
                re[4*j+2] = ((float4c*)data)[j].c.x;
                im[4*j+2] = ((float4c*)data)[j].c.y;
                re[4*j+3] = ((float4c*)data)[j].d.x;
                im[4*j+3] = ((float4c*)data)[j].d.y;
            }
            free(data);
        }
        else if (index->data_type[i] == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double* re = mxGetPr(data_);
            double* im = mxGetPi(data_);
            for (unsigned j = 0; j < m; ++j)
            {
                re[4*j+0] = ((double4c*)data)[j].a.x;
                im[4*j+0] = ((double4c*)data)[j].a.y;
                re[4*j+1] = ((double4c*)data)[j].b.x;
                im[4*j+1] = ((double4c*)data)[j].b.y;
                re[4*j+2] = ((double4c*)data)[j].c.x;
                im[4*j+2] = ((double4c*)data)[j].c.y;
                re[4*j+3] = ((double4c*)data)[j].d.x;
                im[4*j+3] = ((double4c*)data)[j].d.y;
            }
            free(data);
        }
        mxSetField(out[0], i, fields[4], data_);
    }

    oskar_binary_tag_index_free(&index);
    fclose(file);
}


