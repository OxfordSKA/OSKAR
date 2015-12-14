/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_apply_element_taper_cosine.h>
#include <oskar_apply_element_taper_cosine_cuda.h>
#include <oskar_cuda_check_error.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_apply_element_taper_cosine_scalar_f(float2* jones,
        int num_sources, float cos_power, const float* theta)
{
    int i;
    float factor;

    for (i = 0; i < num_sources; ++i)
    {
        /* Compute and apply tapering factor. */
        factor = powf(cosf(theta[i]), cos_power);
        jones[i].x *= factor;
        jones[i].y *= factor;
    }
}

void oskar_apply_element_taper_cosine_matrix_f(float4c* jones,
        int num_sources, float cos_power, const float* theta)
{
    int i;
    float factor;

    for (i = 0; i < num_sources; ++i)
    {
        /* Compute and apply tapering factor. */
        factor = powf(cosf(theta[i]), cos_power);
        jones[i].a.x *= factor;
        jones[i].a.y *= factor;
        jones[i].b.x *= factor;
        jones[i].b.y *= factor;
        jones[i].c.x *= factor;
        jones[i].c.y *= factor;
        jones[i].d.x *= factor;
        jones[i].d.y *= factor;
    }
}

/* Double precision. */
void oskar_apply_element_taper_cosine_scalar_d(double2* jones,
        int num_sources, double cos_power, const double* theta)
{
    int i;
    double factor;

    for (i = 0; i < num_sources; ++i)
    {
        /* Compute and apply tapering factor. */
        factor = pow(cos(theta[i]), cos_power);
        jones[i].x *= factor;
        jones[i].y *= factor;
    }
}

void oskar_apply_element_taper_cosine_matrix_d(double4c* jones,
        int num_sources, double cos_power, const double* theta)
{
    int i;
    double factor;

    for (i = 0; i < num_sources; ++i)
    {
        /* Compute and apply tapering factor. */
        factor = pow(cos(theta[i]), cos_power);
        jones[i].a.x *= factor;
        jones[i].a.y *= factor;
        jones[i].b.x *= factor;
        jones[i].b.y *= factor;
        jones[i].c.x *= factor;
        jones[i].c.y *= factor;
        jones[i].d.x *= factor;
        jones[i].d.y *= factor;
    }
}

/* Wrapper. */
void oskar_apply_element_taper_cosine(oskar_Mem* jones, int num_sources,
        double cos_power, const oskar_Mem* theta, int* status)
{
    int precision, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    precision = oskar_mem_precision(jones);
    type = oskar_mem_type(jones);
    location = oskar_mem_location(jones);

    /* Check arrays are co-located. */
    if (oskar_mem_location(theta) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types for consistency. */
    if (oskar_mem_type(theta) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check precision. */
    if (precision == OSKAR_SINGLE)
    {
        const float* theta_;
        theta_ = oskar_mem_float_const(theta, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (type == OSKAR_SINGLE_COMPLEX)
            {
                oskar_apply_element_taper_cosine_scalar_cuda_f(
                        oskar_mem_float2(jones, status), num_sources,
                        (float)cos_power, theta_);
            }
            else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                oskar_apply_element_taper_cosine_matrix_cuda_f(
                        oskar_mem_float4c(jones, status), num_sources,
                        (float)cos_power, theta_);
            }
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (type == OSKAR_SINGLE_COMPLEX)
            {
                oskar_apply_element_taper_cosine_scalar_f(
                        oskar_mem_float2(jones, status), num_sources,
                        (float)cos_power, theta_);
            }
            else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                oskar_apply_element_taper_cosine_matrix_f(
                        oskar_mem_float4c(jones, status), num_sources,
                        (float)cos_power, theta_);
            }
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else if (precision == OSKAR_DOUBLE)
    {
        const double* theta_;
        theta_ = oskar_mem_double_const(theta, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (type == OSKAR_DOUBLE_COMPLEX)
            {
                oskar_apply_element_taper_cosine_scalar_cuda_d(
                        oskar_mem_double2(jones, status), num_sources,
                        cos_power, theta_);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                oskar_apply_element_taper_cosine_matrix_cuda_d(
                        oskar_mem_double4c(jones, status), num_sources,
                        cos_power, theta_);
            }
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (type == OSKAR_DOUBLE_COMPLEX)
            {
                oskar_apply_element_taper_cosine_scalar_d(
                        oskar_mem_double2(jones, status), num_sources,
                        cos_power, theta_);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                oskar_apply_element_taper_cosine_matrix_d(
                        oskar_mem_double4c(jones, status), num_sources,
                        cos_power, theta_);
            }
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
