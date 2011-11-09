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
#include "math/oskar_jones_set_real_scalar.h"
#include "math/oskar_cuda_jones_set_real_scalar_1.h"
#include "math/oskar_cuda_jones_set_real_scalar_4.h"

extern "C"
int oskar_jones_set_real_scalar(oskar_Jones* jones, double scalar)
{
    // Check that the structure exists.
    if (jones == NULL) return -1;

    // Get the meta-data.
    int n_sources = jones->num_sources();
    int n_stations = jones->num_stations();
    int location = jones->location();
    int type = jones->type();
    int err = 0;

    // Check the location of the data.
    int n_elements = n_sources * n_stations;
    if (location == 0)
    {
        // Data is on the host.
        if (type == OSKAR_SINGLE_COMPLEX)
        {
            float2* ptr = (float2*)jones->ptr.data;
            for (int i = 0; i < n_elements; ++i)
            {
                ptr[i].x = (float)scalar;
                ptr[i].y = 0.0f;
            }
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c* ptr = (float4c*)jones->ptr.data;
            for (int i = 0; i < n_elements; ++i)
            {
                ptr[i].a.x = (float)scalar;
                ptr[i].a.y = 0.0f;
                ptr[i].b.x = 0.0f;
                ptr[i].b.y = 0.0f;
                ptr[i].c.x = 0.0f;
                ptr[i].c.y = 0.0f;
                ptr[i].d.x = (float)scalar;
                ptr[i].d.y = 0.0f;
            }
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            double2* ptr = (double2*)jones->ptr.data;
            for (int i = 0; i < n_elements; ++i)
            {
                ptr[i].x = scalar;
                ptr[i].y = 0.0;
            }
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double4c* ptr = (double4c*)jones->ptr.data;
            for (int i = 0; i < n_elements; ++i)
            {
                ptr[i].a.x = scalar;
                ptr[i].a.y = 0.0;
                ptr[i].b.x = 0.0;
                ptr[i].b.y = 0.0;
                ptr[i].c.x = 0.0;
                ptr[i].c.y = 0.0;
                ptr[i].d.x = scalar;
                ptr[i].d.y = 0.0;
            }
        }
    }
    else if (location == 1)
    {
        // Data is on the device.
        if (type == OSKAR_SINGLE_COMPLEX)
            err = oskar_cuda_jones_set_real_scalar_1_f(n_elements,
                    (float2*)jones->ptr.data, scalar);
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            err = oskar_cuda_jones_set_real_scalar_4_f(n_elements,
                    (float4c*)jones->ptr.data, scalar);
        else if (type == OSKAR_DOUBLE_COMPLEX)
            err = oskar_cuda_jones_set_real_scalar_1_d(n_elements,
                    (double2*)jones->ptr.data, scalar);
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            err = oskar_cuda_jones_set_real_scalar_4_d(n_elements,
                    (double4c*)jones->ptr.data, scalar);
    }

    return err;
}
