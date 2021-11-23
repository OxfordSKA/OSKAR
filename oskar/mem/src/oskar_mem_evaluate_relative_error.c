/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
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
    { \
        /*printf("%.8e, %.8e\n", abs_a, abs_b); */ \
        *max_rel_error = rel_error; \
    } \
    if (min_rel_error && rel_error < *min_rel_error) \
    { \
        *min_rel_error = rel_error; \
    } \
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

/* Switch to absolute error if the absolute difference is less than TOL. */
#define CHECK_ELEMENTS(TOL) \
    for (i = 0; i < n; ++i) \
    { \
        double rel_error = 0.0; \
        const double abs_a = fabs(approx[i]); \
        const double abs_b = fabs(accurate[i]); \
        const double diff = fabs(abs_a - abs_b); \
        if (approx[i] == accurate[i] || \
                (abs_a < FLT_EPSILON && abs_b < FLT_EPSILON)) \
        { \
            rel_error = 0.0; \
        } \
        else if (diff < TOL) \
        { \
            rel_error = diff; \
        } \
        else \
        { \
            rel_error = diff / (abs_a + abs_b); \
        } \
        RUNNING_STATS_KNUTH \
    }

void oskar_mem_evaluate_relative_error(const oskar_Mem* val_approx,
        const oskar_Mem* val_accurate, double* min_rel_error,
        double* max_rel_error, double* avg_rel_error, double* std_rel_error,
        int* status)
{
    int prec_approx = 0, prec_accurate = 0;
    size_t i = 0, n = 0;
    const oskar_Mem *app_ptr = 0, *acc_ptr = 0;
    oskar_Mem *approx_temp = 0, *accurate_temp = 0;
    double old_m = 0.0, new_m = 0.0, old_s = 0.0, new_s = 0.0;
    if (*status) return;

    /* Initialise outputs. */
    if (max_rel_error) *max_rel_error = -DBL_MAX;
    if (min_rel_error) *min_rel_error = DBL_MAX;
    if (avg_rel_error) *avg_rel_error = DBL_MAX;
    if (std_rel_error) *std_rel_error = DBL_MAX;

    /* Type and dimension check. */
    if (oskar_mem_is_matrix(val_approx) && !oskar_mem_is_matrix(val_accurate))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_is_complex(val_approx) && !oskar_mem_is_complex(val_accurate))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get and check base types. */
    prec_approx = oskar_mem_precision(val_approx);
    prec_accurate = oskar_mem_precision(val_accurate);
    if (prec_approx != OSKAR_SINGLE && prec_approx != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (prec_accurate != OSKAR_SINGLE && prec_accurate != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Get number of elements to check. */
    n = oskar_mem_length(val_approx) < oskar_mem_length(val_accurate) ?
            oskar_mem_length(val_approx) : oskar_mem_length(val_accurate);
    if (oskar_mem_is_matrix(val_approx)) n *= 4;

    /* Copy input data to temporary CPU arrays if required. */
    app_ptr = val_approx;
    acc_ptr = val_accurate;
    if (oskar_mem_location(val_approx) != OSKAR_CPU)
    {
        approx_temp = oskar_mem_create_copy(val_approx, OSKAR_CPU,
                status);
        if (*status)
        {
            oskar_mem_free(approx_temp, status);
            return;
        }
        app_ptr = approx_temp;
    }
    if (oskar_mem_location(val_accurate) != OSKAR_CPU)
    {
        accurate_temp = oskar_mem_create_copy(val_accurate, OSKAR_CPU,
                status);
        if (*status)
        {
            oskar_mem_free(accurate_temp, status);
            return;
        }
        acc_ptr = accurate_temp;
    }

    /* Check numbers are the same, to appropriate precision. */
    if (prec_approx == OSKAR_SINGLE && prec_accurate == OSKAR_SINGLE)
    {
        const float *approx = 0, *accurate = 0;
        approx = oskar_mem_float_const(app_ptr, status);
        accurate = oskar_mem_float_const(acc_ptr, status);
        CHECK_ELEMENTS(1e-5)
    }
    else if (prec_approx == OSKAR_DOUBLE && prec_accurate == OSKAR_SINGLE)
    {
        const double *approx = 0;
        const float *accurate = 0;
        approx = oskar_mem_double_const(app_ptr, status);
        accurate = oskar_mem_float_const(acc_ptr, status);
        CHECK_ELEMENTS(1e-5);
    }
    else if (prec_approx == OSKAR_SINGLE && prec_accurate == OSKAR_DOUBLE)
    {
        const float *approx = 0;
        const double *accurate = 0;
        approx = oskar_mem_float_const(app_ptr, status);
        accurate = oskar_mem_double_const(acc_ptr, status);
        CHECK_ELEMENTS(1e-5);
    }
    else if (prec_approx == OSKAR_DOUBLE && prec_accurate == OSKAR_DOUBLE)
    {
        const double *approx = 0, *accurate = 0;
        approx = oskar_mem_double_const(app_ptr, status);
        accurate = oskar_mem_double_const(acc_ptr, status);
        CHECK_ELEMENTS(1e-15);
    }

    /* Set mean and standard deviation of relative error. */
    if (avg_rel_error) *avg_rel_error = new_m;
    if (std_rel_error) *std_rel_error = (n > 0) ? sqrt(new_s / n) : 0.0;

    /* Clean up temporaries if required. */
    oskar_mem_free(approx_temp, status);
    oskar_mem_free(accurate_temp, status);
}

#ifdef __cplusplus
}
#endif
