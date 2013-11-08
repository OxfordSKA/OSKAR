/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_convert_relative_direction_cosines_to_enu_direction_cosines.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int np, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, double ra0, double dec0,
        double LAST, double lat, int* status)
{
    int type; /* precision of memory arrays */

    if (!status || *status != OSKAR_SUCCESS)
        return;
    if (!x || !y || !z || !l || !m || !n) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Obtain memory type and check consistency */
    if (oskar_mem_type(x) == OSKAR_DOUBLE &&
        oskar_mem_type(y) == OSKAR_DOUBLE &&
        oskar_mem_type(z) == OSKAR_DOUBLE &&
        oskar_mem_type(l) == OSKAR_DOUBLE &&
        oskar_mem_type(m) == OSKAR_DOUBLE &&
        oskar_mem_type(n) == OSKAR_DOUBLE)
    {
        type = OSKAR_DOUBLE;
    }
    else if (oskar_mem_type(x) == OSKAR_SINGLE &&
             oskar_mem_type(y) == OSKAR_SINGLE &&
             oskar_mem_type(z) == OSKAR_SINGLE &&
             oskar_mem_type(l) == OSKAR_SINGLE &&
             oskar_mem_type(m) == OSKAR_SINGLE &&
             oskar_mem_type(n) == OSKAR_SINGLE)
    {
        type = OSKAR_SINGLE;
    }
    else
    {
        /* TODO new error code? The text string for this one is:
         * "unsupported data type" and should be more along the lines of
         * "mismatched data type"
         */
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check memory arrays for dimension consistency */
    if ((int)oskar_mem_length(x) < np ||
        (int)oskar_mem_length(y) < np ||
        (int)oskar_mem_length(z) < np ||
        (int)oskar_mem_length(l) < np ||
        (int)oskar_mem_length(m) < np ||
        (int)oskar_mem_length(n) < np)
    {
        printf("length(x,y,z) = (%i,%i,%i)\n", oskar_mem_length(x),
                oskar_mem_length(y),oskar_mem_length(z));
        printf("length(l,m,n) = (%i,%i,%i)\n", oskar_mem_length(l),
                oskar_mem_length(n),oskar_mem_length(m));
        fflush(stdout);
        printf("STATUS = %i\n", *status); fflush(stdout);
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        printf("STATUS = %i\n", *status); fflush(stdout);
        return;
    }


    /* All OSKAR memory structure are on the CPU */
    if (oskar_mem_location(x) == OSKAR_LOCATION_CPU &&
        oskar_mem_location(y) == OSKAR_LOCATION_CPU &&
        oskar_mem_location(z) == OSKAR_LOCATION_CPU &&
        oskar_mem_location(l) == OSKAR_LOCATION_CPU &&
        oskar_mem_location(m) == OSKAR_LOCATION_CPU &&
        oskar_mem_location(n) == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE) {
            oskar_convert_relative_direction_cosines_to_enu_direction_cosines_d(
                    oskar_mem_double(x, status), oskar_mem_double(y, status),
                    oskar_mem_double(z, status), np,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status), ra0, dec0, LAST, lat);
        }
        else {
            oskar_convert_relative_direction_cosines_to_enu_direction_cosines_f(
                    oskar_mem_float(x, status), oskar_mem_float(y, status),
                    oskar_mem_float(z, status), np,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status), ra0, dec0, LAST, lat);
        }
    }
    /* ALl OSKAR memory structures are on the GPU */
    else if (oskar_mem_location(x) == OSKAR_LOCATION_GPU &&
            oskar_mem_location(y) == OSKAR_LOCATION_GPU &&
            oskar_mem_location(z) == OSKAR_LOCATION_GPU &&
            oskar_mem_location(l) == OSKAR_LOCATION_GPU &&
            oskar_mem_location(m) == OSKAR_LOCATION_GPU &&
            oskar_mem_location(n) == OSKAR_LOCATION_GPU)
    {
        /* TODO CUDA kernels */
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
}

void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_d(
        double* x, double* y, double* z, int np, const double* l, const double* m,
        const double* n, double ra0, double dec0, double LAST, double lat)
{
    int i;
    double sinDec0, cosDec0, sinLat, cosLat;
    sinDec0 = sin(dec0);
    cosDec0 = cos(dec0);
    sinLat = sin(lat);
    cosLat = cos(lat);

    #pragma omp parallel for
    for ( i = 0; i < np; ++i)
    {
        double ra, ha, dec, sinHA, cosHA, sinDec, cosDec, tmp;

        /* relative l,m,n to ha, dec */
        dec = asin(n[i] * sinDec0 + m[i] * cosDec0);
        ra = (ra0 + atan2(l[i], cosDec0 * n[i] - m[i] * sinDec0));
        ha  = LAST - ra;

        /* ha, dec to enu directions */
        sinHA  = sin(ha);
        cosHA  = cos(ha);
        sinDec = sin(dec);
        cosDec = cos(dec);

        tmp  = cosDec * cosHA;

        x[i] = -cosDec * sinHA;
        y[i] = cosLat * sinDec - sinLat * tmp;
        z[i] = sinLat * sinDec + cosLat * tmp;
    }


#if 0
    /* This function is currently a combination of: */
    /*      relative_direction_cosines_to_apparent_ra_dec (ha_dec) */
    /*      apparent_ra_dec_to_enu_direction_cosines */
    /* TODO do this as a rotation instead. */

    int i;
    double sinDec0, cosDec0, sinLat, cosLat;
    sinDec0 = sin(dec0);
    cosDec0 = cos(dec0);
    sinLat = sin(lat);
    cosLat = cos(lat);

    #pragma omp parallel for
    for ( i = 0; i < np; ++i)
    {
        double ha, dec, sinHA, cosHA, sinDec, cosDec, tmp;

        /* relative l,m,n to ha, dec */
        dec = asin(n[i] * sinDec0 + m[i] * cosDec0);
        ha  = LAST - (ra0 + atan2(l[i], cosDec0 * n[i] - m[i] * sinDec0));

        /* ha, dec to enu directions */
        sinHA  = sin(ha);
        cosHA  = cos(ha);
        sinDec = sin(dec);
        cosDec = cos(dec);

        tmp  = cosDec * cosHA;

        x[i] = -cosDec * sinHA;
        y[i] = cosLat * sinDec - sinLat * tmp;
        z[i] = sinLat * sinDec + cosLat * tmp;
    }
#endif
}

void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_f(
        float* x, float* y, float* z, int np, const float* l, const float* m,
        const float* n, float ra0, float dec0, float LAST, float lat)
{
    /* This function is currently a combination of: */
    /*      relative_direction_cosines_to_apparent_ra_dec (ha_dec) */
    /*      apparent_ra_dec_to_enu_direction_cosines */
    /* TODO do this as a rotation instead. */

    int i;
    float sinDec0, cosDec0, sinLat, cosLat;
    sinDec0 = sinf(dec0);
    cosDec0 = cosf(dec0);
    sinLat = sinf(lat);
    cosLat = cosf(lat);

    #pragma omp parallel for
    for ( i = 0; i < np; ++i)
    {
        float ha, dec, sinHA, cosHA, sinDec, cosDec, tmp;

        /* relative l,m,n to ha, dec */
        dec = asinf(n[i] * sinDec0 + m[i] * cosDec0);
        ha  = LAST - (ra0 + atan2f(l[i], cosDec0 * n[i] - m[i] * sinDec0));

        /* ha, dec to enu directions */
        sinHA  = sinf(ha);
        cosHA  = cosf(ha);
        sinDec = sinf(dec);
        cosDec = cosf(dec);

        tmp  = cosDec * cosHA;

        x[i] = -cosDec * sinHA;
        y[i] = cosLat * sinDec - sinLat * tmp;
        z[i] = sinLat * sinDec + cosLat * tmp;
    }
}

#ifdef __cplusplus
}
#endif
