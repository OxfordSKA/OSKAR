/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_convert_theta_phi_to_enu_direction_cosines.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_theta_phi_to_enu_direction_cosines(oskar_Mem* x,
        oskar_Mem* y, oskar_Mem* z, unsigned int np, const oskar_Mem* theta,
        const oskar_Mem* phi, int* status)
{
    int type, loc;

    if (!status || *status != OSKAR_SUCCESS)
        return;

    if (!x || !y || !z || !theta || !phi)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    type = oskar_mem_type(x);
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(theta) != type || oskar_mem_type(phi) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    loc = oskar_mem_location(x);
    if (oskar_mem_location(y) != loc || oskar_mem_location(z) != loc ||
            oskar_mem_location(theta) != loc || oskar_mem_location(phi) != loc)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    if (loc != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    if (oskar_mem_length(x) < np || oskar_mem_length(y) < np ||
            oskar_mem_length(z) < np || oskar_mem_length(theta) < np ||
            oskar_mem_length(phi) < np)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    if (type == OSKAR_DOUBLE)
    {
        oskar_convert_theta_phi_to_enu_direction_cosines_d(
                oskar_mem_double(x, status), oskar_mem_double(y, status),
                oskar_mem_double(z, status), np,
                oskar_mem_double_const(theta, status),
                oskar_mem_double_const(phi, status));
    }
    else if (type == OSKAR_SINGLE)
    {
        oskar_convert_theta_phi_to_enu_direction_cosines_f(
                oskar_mem_float(x, status), oskar_mem_float(y, status),
                oskar_mem_float(z, status),
                np, oskar_mem_float_const(theta, status),
                oskar_mem_float_const(phi, status));
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
}


void oskar_convert_theta_phi_to_enu_direction_cosines_d(double* x,
        double* y, double* z, unsigned int np, const double* theta, const double* phi)
{
    unsigned int i;
    double sinTheta;
    for (i = 0; i < np; ++i)
    {
        sinTheta = sin(theta[i]);
        x[i] = sinTheta * cos(phi[i]);
        y[i] = sinTheta * sin(phi[i]);
        z[i] = cos(theta[i]);
    }
}

void oskar_convert_theta_phi_to_enu_direction_cosines_f(float* x,
        float* y, float* z, unsigned int np, const float* theta, const float* phi)
{
    unsigned int i;
    float sinTheta;
    for (i = 0; i < np; ++i)
    {
        sinTheta = sinf(theta[i]);
        x[i] = sinTheta * cosf(phi[i]);
        y[i] = sinTheta * sinf(phi[i]);
        z[i] = cosf(theta[i]);
    }
}


#ifdef __cplusplus
}
#endif
