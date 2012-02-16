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

#include <cuda_runtime_api.h>
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_set_value_real(oskar_Mem* mem, double val)
{
    int i, n, type, location;

    /* Get the data type, location, and number of elements. */
    type = mem->private_type;
    location = mem->private_location;
    n = mem->private_num_elements;

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            double *v;
            v = (double*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = val;
        }
        else if (type == OSKAR_SINGLE)
        {
            float *v;
            v = (float*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = val;
        }
        else
            return OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            double *v;
            v = (double*)(mem->data);
            for (i = 0; i < n; ++i)
                cudaMemcpy(v + i, &val, sizeof(double), cudaMemcpyHostToDevice);
        }
        else if (type == OSKAR_SINGLE)
        {
            float t, *v;
            t = (float) val;
            v = (float*)(mem->data);
            for (i = 0; i < n; ++i)
                cudaMemcpy(v + i, &t, sizeof(float), cudaMemcpyHostToDevice);
        }
        else
            return OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
        return OSKAR_ERR_BAD_LOCATION;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
