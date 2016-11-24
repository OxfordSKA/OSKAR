/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "math/oskar_fit_ellipse.h"
#include "math/oskar_matrix_multiply.h"
#include "math/oskar_lapack_subset.h"

#include "math/oskar_cmath.h"
#include <string.h> /* For memset() */

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_fit_ellipse_f(float* major, float* minor,
        float* position_angle_rad, int num_points, const float* x,
        const float* y, float* work1_X, float* work2_XX, int* status)
{
    int i, j;
    float a, b, c, d, e, mean_x = 0.0, mean_y = 0.0, sumX[5];

    if (num_points < 5)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Clear arrays. */
    memset(work2_XX, 0, 5 * num_points * sizeof(float));
    memset(sumX, 0, sizeof(sumX));

    /* Estimate the conic equation of the ellipse, first removing the mean
     * of the input data to improve the accuracy of matrix inversion. */
    for (i = 0; i < num_points; ++i)
    {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= (float)num_points;
    mean_y /= (float)num_points;

    for (i = 0; i < num_points; ++i)
    {
        float x_, y_, t[5];

        /* Construct a row in the temporary matrix. */
        x_ = x[i] - mean_x;
        y_ = y[i] - mean_y;
        t[0] = x_ * x_;
        t[1] = x_ * y_;
        t[2] = y_ * y_;
        t[3] = x_;
        t[4] = y_;

        /* Fill in the row and increment the sum for each column. */
        for (j = 0; j < 5; ++j)
        {
            work1_X[5 * i + j] = t[j];
            sumX[j] += t[j];
        }
    }

    /* Result = sum(X) / (X' * X) */
    oskar_matrix_multiply_f(work2_XX, num_points, 5, num_points, 5,
            OSKAR_TRUE, OSKAR_FALSE, work1_X, work1_X, status);
    if (*status) return;

    {
        int ipiv[5], n, m = 5, lda = 5, ldb = 5, info = 0, nrhs = 1;
        char trans = 'N';
        /* Solve system of linear equations using LU factorisation in XX. */
        memset(ipiv, 0, sizeof(ipiv));
        n = num_points;
        oskar_sgetrf(m, n, work2_XX, lda, ipiv, &info); /* LU factorisation. */
        if (info != 0)
            *status = OSKAR_ERR_ELLIPSE_FIT_FAILED;
        oskar_sgetrs(&trans, m, nrhs, work2_XX, lda, ipiv, sumX, ldb);
    }

    /* Extract resulting conic equation parameters from sumX. */
    a = sumX[0]; b = sumX[1]; c = sumX[2]; d = sumX[3]; e = sumX[4];

    /* Remove the orientation from the ellipse. */
    if (MIN(fabs(b / a), fabs(b / c)) > 1e-3 /* orientation tolerance */)
    {
        float cos_phi, sin_phi, a2, c2, d2, e2;
        *position_angle_rad = 0.5 * atan2(b, (c - a)) - M_PI / 2.0;
        cos_phi = cos(*position_angle_rad);
        sin_phi = sin(*position_angle_rad);
        a2 = a*cos_phi*cos_phi - b*cos_phi*sin_phi + c*sin_phi*sin_phi;
        c2 = a*sin_phi*sin_phi + b*cos_phi*sin_phi + c*cos_phi*cos_phi;
        d2 = d*cos_phi - e*sin_phi;
        e2 = d*sin_phi + e*cos_phi;
        a = a2; c = c2; d = d2; e = e2;
    }
    else
    {
        *position_angle_rad = (c - a < 0.0) ? 0.0 : M_PI / 2.0;
    }

    /* Check if conic equation represents an ellipse. */
    if (a * c <= 0.0)
    {
        *status = OSKAR_ERR_ELLIPSE_FIT_FAILED;
    }
    else
    {
        float F;
        if (a < 0.0)
        {
            a = -a;
            c = -c;
            d = -d;
            e = -e;
        }

        /* Major and minor axes. */
        F = 1.0 + (d * d) / (4.0 * a) + (e * e) / (4.0 * c);
        a = sqrt(F / a);
        b = sqrt(F / c);
        *major = 2.0 * MAX(a, b);
        *minor = 2.0 * MIN(a, b);
    }
}


/* Double precision. */
void oskar_fit_ellipse_d(double* major, double* minor,
        double* position_angle_rad, int num_points, const double* x,
        const double* y, double* work1_X, double* work2_XX, int* status)
{
    int i, j;
    double a, b, c, d, e, mean_x = 0.0, mean_y = 0.0, sumX[5];

    if (num_points < 5)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Clear arrays. */
    memset(work2_XX, 0, 5 * num_points * sizeof(double));
    memset(sumX, 0, sizeof(sumX));

    /* Estimate the conic equation of the ellipse, first removing the mean
     * of the input data to improve the accuracy of matrix inversion. */
    for (i = 0; i < num_points; ++i)
    {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= (double)num_points;
    mean_y /= (double)num_points;

    for (i = 0; i < num_points; ++i)
    {
        double x_, y_, t[5];

        /* Construct a row in the temporary matrix. */
        x_ = x[i] - mean_x;
        y_ = y[i] - mean_y;
        t[0] = x_ * x_;
        t[1] = x_ * y_;
        t[2] = y_ * y_;
        t[3] = x_;
        t[4] = y_;

        /* Fill in the row and increment the sum for each column. */
        for (j = 0; j < 5; ++j)
        {
            work1_X[5 * i + j] = t[j];
            sumX[j] += t[j];
        }
    }

    /* Result = sum(X) / (X' * X) */
    oskar_matrix_multiply_d(work2_XX, num_points, 5, num_points, 5,
            OSKAR_TRUE, OSKAR_FALSE, work1_X, work1_X, status);
    if (*status) return;

    {
        int ipiv[5], n, m = 5, lda = 5, ldb = 5, info = 0, nrhs = 1;
        char trans = 'N';
        /* Solve system of linear equations using LU factorisation in XX. */
        memset(ipiv, 0, sizeof(ipiv));
        n = num_points;
        oskar_dgetrf(m, n, work2_XX, lda, ipiv, &info); /* LU factorisation. */
        if (info != 0)
            *status = OSKAR_ERR_ELLIPSE_FIT_FAILED;
        oskar_dgetrs(&trans, m, nrhs, work2_XX, lda, ipiv, sumX, ldb);
    }

    /* Extract resulting conic equation parameters from sumX. */
    a = sumX[0]; b = sumX[1]; c = sumX[2]; d = sumX[3]; e = sumX[4];

    /* Remove the orientation from the ellipse. */
    if (MIN(fabs(b / a), fabs(b / c)) > 1e-3 /* orientation tolerance */)
    {
        double cos_phi, sin_phi, a2, c2, d2, e2;
        *position_angle_rad = 0.5 * atan2(b, (c - a)) - M_PI / 2.0;
        cos_phi = cos(*position_angle_rad);
        sin_phi = sin(*position_angle_rad);
        a2 = a*cos_phi*cos_phi - b*cos_phi*sin_phi + c*sin_phi*sin_phi;
        c2 = a*sin_phi*sin_phi + b*cos_phi*sin_phi + c*cos_phi*cos_phi;
        d2 = d*cos_phi - e*sin_phi;
        e2 = d*sin_phi + e*cos_phi;
        a = a2; c = c2; d = d2; e = e2;
    }
    else
    {
        *position_angle_rad = (c - a < 0.0) ? 0.0 : M_PI / 2.0;
    }

    /* Check if conic equation represents an ellipse. */
    if (a * c <= 0.0)
    {
        *status = OSKAR_ERR_ELLIPSE_FIT_FAILED;
    }
    else
    {
        double F;
        if (a < 0.0)
        {
            a = -a;
            c = -c;
            d = -d;
            e = -e;
        }

        /* Major and minor axes. */
        F = 1.0 + (d * d) / (4.0 * a) + (e * e) / (4.0 * c);
        a = sqrt(F / a);
        b = sqrt(F / c);
        *major = 2.0 * MAX(a, b);
        *minor = 2.0 * MIN(a, b);
    }
}

#ifdef __cplusplus
}
#endif
