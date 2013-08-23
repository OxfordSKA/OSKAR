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

#include <oskar_mem_copy.h>
#include <oskar_mem_evaluate_relative_error.h>
#include <oskar_mem_free.h>
#include <oskar_mem_init.h>
#include <oskar_mem_type_check.h>
#include <oskar_vector_types.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Macro to update running statistics for mean and standard deviation using
 * the method of Donald Knuth in "The Art of Computer Programming"
 * vol 2, 3rd edition, page 232 */
#define RUNNING_STATS_KNUTH \
    if (max_rel_error && rel_error > *max_rel_error) \
        *max_rel_error = rel_error; \
    if (min_rel_error && rel_error < *min_rel_error) \
        *min_rel_error = rel_error; \
    if (i == 0) \
    { \
        old_m = new_m = rel_error; \
        old_s = 0.0; \
    } \
    else \
    { \
        new_m = old_m + (rel_error - old_m) / (i + 1); \
        new_s = old_s + (rel_error - old_m) * (rel_error - new_m); \
        old_m = new_m; \
        old_s = new_s; \
    }

#define CHECK_ELEMENTS_SCALAR \
    for (i = 0; i < n; ++i) \
    { \
        double rel_error; \
        rel_error = fabs(approx[i] / accurate[i] - 1.0); \
        if (approx[i] == 0.0 || accurate[i] == 0.0) \
            rel_error = 0.0; \
        RUNNING_STATS_KNUTH \
    }

#define CHECK_ELEMENTS_COMPLEX \
    for (i = 0; i < n; ++i) \
    { \
        double a_approx, a_accurate, rel_error; \
        a_approx = sqrt(pow(approx[i].x, 2.0) + pow(approx[i].y, 2.0)); \
        a_accurate = sqrt(pow(accurate[i].x, 2.0) + pow(accurate[i].y, 2.0)); \
        rel_error = fabs(a_approx / a_accurate - 1.0); \
        if (a_approx == 0.0 || a_accurate == 0.0) \
            rel_error = 0.0; \
        RUNNING_STATS_KNUTH \
    }


void oskar_mem_evaluate_relative_error(const oskar_Mem* val_approx,
        const oskar_Mem* val_accurate, double* min_rel_error,
        double* max_rel_error, double* avg_rel_error, double* std_rel_error,
        int* status)
{
    int base_type_approx, base_type_accurate, i, n;
    const oskar_Mem *app_ptr, *acc_ptr;
    oskar_Mem approx_temp, accurate_temp;
    double old_m = 0.0, new_m = 0.0, old_s = 0.0, new_s = 0.0;

    /* Check all inputs. */
    if (!val_approx || !val_accurate || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Initialise outputs. */
    if (max_rel_error) *max_rel_error = -DBL_MAX;
    if (min_rel_error) *min_rel_error = DBL_MAX;
    if (avg_rel_error) *avg_rel_error = DBL_MAX;
    if (std_rel_error) *std_rel_error = DBL_MAX;

    /* Type and dimension check. */
    if (oskar_mem_is_matrix(val_approx->type) &&
            !oskar_mem_is_matrix(val_accurate->type))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_is_complex(val_approx->type) &&
            !oskar_mem_is_complex(val_accurate->type))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get and check base types. */
    base_type_approx = oskar_mem_base_type(val_approx->type);
    base_type_accurate = oskar_mem_base_type(val_accurate->type);
    if (base_type_approx != OSKAR_SINGLE &&
            base_type_approx != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (base_type_accurate != OSKAR_SINGLE &&
            base_type_accurate != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Get number of elements to check. */
    n = val_approx->num_elements < val_accurate->num_elements ?
            val_approx->num_elements : val_accurate->num_elements;
    if (oskar_mem_is_matrix(val_approx->type)) n *= 4;

    /* Copy input data to temporary CPU arrays if required. */
    app_ptr = val_approx;
    acc_ptr = val_accurate;
    if (val_approx->location != OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&approx_temp, val_approx->type, OSKAR_LOCATION_CPU,
                0, 1, status);
        oskar_mem_copy(&approx_temp, val_approx, status);
        app_ptr = &approx_temp;
    }
    if (val_accurate->location != OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&accurate_temp, val_accurate->type, OSKAR_LOCATION_CPU,
                0, 1, status);
        oskar_mem_copy(&accurate_temp, val_accurate, status);
        acc_ptr = &accurate_temp;
    }
    if (*status)
    {
        oskar_mem_free(&approx_temp, status);
        oskar_mem_free(&accurate_temp, status);
        return;
    }

    /* Check numbers are the same, to appropriate precision. */
    if (oskar_mem_is_complex(val_approx->type))
    {
        if (base_type_approx == OSKAR_SINGLE &&
                base_type_accurate == OSKAR_SINGLE)
        {
            const float2 *approx, *accurate;
            approx = (const float2*)(app_ptr->data);
            accurate = (const float2*)(acc_ptr->data);
            CHECK_ELEMENTS_COMPLEX
        }
        else if (base_type_approx == OSKAR_DOUBLE &&
                base_type_accurate == OSKAR_SINGLE)
        {
            const double2 *approx;
            const float2 *accurate;
            approx = (const double2*)(app_ptr->data);
            accurate = (const float2*)(acc_ptr->data);
            CHECK_ELEMENTS_COMPLEX
        }
        else if (base_type_approx == OSKAR_SINGLE &&
                base_type_accurate == OSKAR_DOUBLE)
        {
            const float2 *approx;
            const double2 *accurate;
            approx = (const float2*)(app_ptr->data);
            accurate = (const double2*)(acc_ptr->data);
            CHECK_ELEMENTS_COMPLEX
        }
        else if (base_type_approx == OSKAR_DOUBLE &&
                base_type_accurate == OSKAR_DOUBLE)
        {
            const double2 *approx, *accurate;
            approx = (const double2*)(app_ptr->data);
            accurate = (const double2*)(acc_ptr->data);
            CHECK_ELEMENTS_COMPLEX
        }
    }
    else
    {
        if (base_type_approx == OSKAR_SINGLE &&
                base_type_accurate == OSKAR_SINGLE)
        {
            const float *approx, *accurate;
            approx = (const float*)(app_ptr->data);
            accurate = (const float*)(acc_ptr->data);
            CHECK_ELEMENTS_SCALAR
        }
        else if (base_type_approx == OSKAR_DOUBLE &&
                base_type_accurate == OSKAR_SINGLE)
        {
            const double *approx;
            const float *accurate;
            approx = (const double*)(app_ptr->data);
            accurate = (const float*)(acc_ptr->data);
            CHECK_ELEMENTS_SCALAR
        }
        else if (base_type_approx == OSKAR_SINGLE &&
                base_type_accurate == OSKAR_DOUBLE)
        {
            const float *approx;
            const double *accurate;
            approx = (const float*)(app_ptr->data);
            accurate = (const double*)(acc_ptr->data);
            CHECK_ELEMENTS_SCALAR
        }
        else if (base_type_approx == OSKAR_DOUBLE &&
                base_type_accurate == OSKAR_DOUBLE)
        {
            const double *approx, *accurate;
            approx = (const double*)(app_ptr->data);
            accurate = (const double*)(acc_ptr->data);
            CHECK_ELEMENTS_SCALAR
        }
    }

    /* Set mean and standard deviation of relative error. */
    if (avg_rel_error) *avg_rel_error = new_m;
    if (std_rel_error) *std_rel_error = (n > 1) ? sqrt(new_s / (n - 1)) : 0.0;

    /* Clean up temporaries if required. */
    if (val_approx->location != OSKAR_LOCATION_CPU)
        oskar_mem_free(&approx_temp, status);
    if (val_accurate->location != OSKAR_LOCATION_CPU)
        oskar_mem_free(&accurate_temp, status);
}

#ifdef __cplusplus
}
#endif
