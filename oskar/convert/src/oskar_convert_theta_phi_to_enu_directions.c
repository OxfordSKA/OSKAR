/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_theta_phi_to_enu_directions.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_theta_phi_to_enu_directions(unsigned int num,
        const oskar_Mem* theta, const oskar_Mem* phi, oskar_Mem* x,
        oskar_Mem* y, oskar_Mem* z, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(theta) != type || oskar_mem_type(phi) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(theta) != location ||
            oskar_mem_location(phi) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (oskar_mem_length(x) < num || oskar_mem_length(y) < num ||
            oskar_mem_length(z) < num || oskar_mem_length(theta) < num ||
            oskar_mem_length(phi) < num)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (type == OSKAR_DOUBLE)
    {
        oskar_convert_theta_phi_to_enu_directions_d(num,
                oskar_mem_double_const(theta, status),
                oskar_mem_double_const(phi, status),
                oskar_mem_double(x, status), oskar_mem_double(y, status),
                oskar_mem_double(z, status));
    }
    else if (type == OSKAR_SINGLE)
    {
        oskar_convert_theta_phi_to_enu_directions_f(num,
                oskar_mem_float_const(theta, status),
                oskar_mem_float_const(phi, status),
                oskar_mem_float(x, status), oskar_mem_float(y, status),
                oskar_mem_float(z, status));
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
}

void oskar_convert_theta_phi_to_enu_directions_d(unsigned int num,
        const double* theta, const double* phi, double* x, double* y,
        double* z)
{
    unsigned int i = 0;
    for (i = 0; i < num; ++i)
    {
        const double sin_theta = sin(theta[i]);
        x[i] = sin_theta * cos(phi[i]);
        y[i] = sin_theta * sin(phi[i]);
        z[i] = cos(theta[i]);
    }
}

void oskar_convert_theta_phi_to_enu_directions_f(unsigned int num,
        const float* theta, const float* phi, float* x, float* y,
        float* z)
{
    unsigned int i = 0;
    for (i = 0; i < num; ++i)
    {
        const float sin_theta = sinf(theta[i]);
        x[i] = sin_theta * cosf(phi[i]);
        y[i] = sin_theta * sinf(phi[i]);
        z[i] = cosf(theta[i]);
    }
}

#ifdef __cplusplus
}
#endif
