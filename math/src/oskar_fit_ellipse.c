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


#include "math/oskar_fit_ellipse.h"
#include "math/oskar_mean.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "math/oskar_matrix_multiply.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OSKAR_NO_LAPACK
/* http://www.netlib.org/lapack/double/dgesv.f */
extern void dgesv_(const int* n, const int* nrhs, double* A, const int lda,
        int* ipiv, double* B, const int* ldb, int* info);
extern void sgesv_(const int* n, const int* nrhs, double* A, const int lda,
        int* ipiv, double* B, const int* ldb, int* info);


/* http://www.netlib.org/lapack/double/dgetrf.f */
extern void dgetrf_(const int* m, const int* n, double* A, const int* lda,
        int* ipiv, int* info);
extern void sgetrf_(const int* m, const int* n, float* A, const int* lda,
        int* ipiv, int* info);

/* http://www.netlib.org/lapack/double/dgetrs.f */
extern void dgetrs_(const char* trans, const int* n, const int* nrhs,
        double* A, const int* lda, int* ipiv, double* B, const int* lba,
        int* info);
extern void sgetrs_(const char* trans, const int* n, const int* nrhs,
        float* A, const int* lda, int* ipiv, float* B, const int* lba,
        int* info);
#endif




int oskar_fit_ellipse(double* gauss_maj, double* gauss_min,
        double* gauss_phi, int num_points, const oskar_Mem* x,
        const oskar_Mem* y)
{
    int i, j, type, location, err, rows_X, cols_X, n, m, lda, ldb, info, nrhs;
    double orientation_tolerance, orientation_rad, cos_phi, sin_phi;
    double a1, b1, c1, d1, e1;
    double a2, b2, c2, d2, e2;
    double mean_x_1, mean_y_1;
    double mean_x_2, mean_y_2;
    double test;
    double F;
    oskar_Mem x2, y2, X, XX, sumX, ipiv;
    char trans;
    err = OSKAR_SUCCESS;

    if (num_points < 5)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

#ifdef OSKAR_NO_LAPACK
    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE)
        type = OSKAR_DOUBLE;
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE)
        type = OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    location = OSKAR_LOCATION_CPU;

    err = oskar_mem_init(&x2, type, location, num_points, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&y2, type, location, num_points, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&X, type, location, 5 * num_points, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&XX, type, location, 5 * num_points, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&sumX, type, location, 5, OSKAR_TRUE);
    if (err) return err;

    /* remove bias of ellipse */
    err = oskar_mean(&mean_x_1, num_points, x);
    if (err) return err;
    err = oskar_mean(&mean_y_1, num_points, y);
    if (err) return err;
    if (type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_points; ++i)
        {
            ((double*)x2.data)[i] = ((double*)x->data)[i] - mean_x_1;
            ((double*)y2.data)[i] = ((double*)y->data)[i] - mean_y_1;
        }
    }
    else
    {
        for (i = 0; i < num_points; ++i)
        {
            ((float*)x2.data)[i] = ((float*)x->data)[i] - mean_x_1;
            ((float*)y2.data)[i] = ((float*)y->data)[i] - mean_y_1;
        }
    }

    /* estimation of the conic equation of the ellipse */
    if (type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_points; ++i)
        {
            double x_, y_;
            x_ = ((double*)x2.data)[i];
            y_ = ((double*)y2.data)[i];
            ((double*)X.data)[i * 5 + 0] = x_ * x_;
            ((double*)X.data)[i * 5 + 1] = x_ * y_;
            ((double*)X.data)[i * 5 + 2] = y_ * y_;
            ((double*)X.data)[i * 5 + 3] = x_;
            ((double*)X.data)[i * 5 + 4] = y_;
        }
    }
    else
    {
        for (i = 0; i < num_points; ++i)
        {
            float x_, y_;
            x_ = ((float*)x2.data)[i];
            y_ = ((float*)y2.data)[i];
            ((float*)X.data)[i * 5 + 0] = x_ * x_;
            ((float*)X.data)[i * 5 + 1] = x_ * y_;
            ((float*)X.data)[i * 5 + 2] = y_ * y_;
            ((float*)X.data)[i * 5 + 3] = x_;
            ((float*)X.data)[i * 5 + 4] = y_;
        }
    }

    /*
        printf("\n");
        for (j = 0; j < num_points; ++j)
        {
            for (i = 0; i < 5; ++i)
            {
                printf("% -.4f ", ((double*)X.data)[j * 5 + i]);
            }
            printf("\n");
        }
     */


    /* TODO result = sum(X) / (X' * X) */
    cols_X = 5;
    rows_X = num_points;

    err = oskar_matrix_multiply(&XX, rows_X, cols_X, rows_X, cols_X, OSKAR_TRUE,
            OSKAR_FALSE, &X, &X);
    if (err) return err;

    /*
        printf("\n");
        for (j = 0; j < 5; ++j)
        {
            for (i = 0; i < 5; ++i)
            {
                printf("% -.4f ", ((double*)XX.data)[j * 5 + i]);
            }
            printf("\n");
        }
     */

    if (type == OSKAR_DOUBLE)
    {
        for (j = 0; j < rows_X; ++j)
        {
            for (i = 0; i < cols_X; ++i)
            {
                ((double*)sumX.data)[i] += ((double*)X.data)[j * cols_X + i];
            }
        }
    }
    else
    {
        for (j = 0; j < rows_X; ++j)
        {
            for (i = 0; i < cols_X; ++i)
            {
                ((float*)sumX.data)[i] += ((float*)X.data)[j * cols_X + i];
            }
        }
    }

    /*
        printf("\n");
        for (j = 0; j < 5; ++j)
        {
            printf("%f ", ((double*)sumX.data)[j]);
        }
        printf("\n");
     */


    info = 0;
    n = rows_X;
    m = cols_X;
    lda = MAX(1, m);

    err = oskar_mem_init(&ipiv, OSKAR_INT, location, MIN(m, n), OSKAR_TRUE);
    if (err) return err;

#ifndef OSKAR_NO_LAPACK
    if (type == OSKAR_DOUBLE)
    {
        dgetrf_(&m, &n, (double*)XX.data, &lda, (int*)ipiv.data, &info);
        if (info != 0) return info;

        n = cols_X;
        ldb = MAX(1, n);
        nrhs = 1;
        trans = 'N';
        dgetrs_(&trans, &n, &nrhs, (double*)XX.data, &lda, (int*)ipiv.data,
                (double*)sumX.data, &ldb, &info);
        if (info != 0) return info;
    }
    else
    {
        sgetrf_(&m, &n, (float*)XX.data, &lda, (int*)ipiv.data, &info);
        if (info != 0) return OSKAR_ERR_UNKNOWN;

        n = cols_X;
        ldb = MAX(1, n);
        nrhs = 1;
        trans = 'N';
        sgetrs_(&trans, &n, &nrhs, (float*)XX.data, &lda, (int*)ipiv.data,
                (float*)sumX.data, &ldb, &info);
        if (info != 0) return OSKAR_ERR_UNKNOWN;
    }
#endif

//#ifndef OSKAR_NO_LAPACK
//    if (type == OSKAR_DOUBLE)
//    {
//        dgesv_(&num_points, &cols_X, (double*)XX.data, &lda, (int*)ipiv.data, &info);
//        if (info != 0) return OSKAR_ERR_UNKNOWN;
//
//    }
//    else
//    {
//
//    }
//#endif

    /*
        printf("\n");
        for (j = 0; j < 5; ++j)
        {
            printf("%f ", ((double*)sumX.data)[j]);
        }
        printf("\n");
     */

    /* extract parameters from conic equation */
    if (type == OSKAR_DOUBLE)
    {
        a1 = ((double*)sumX.data)[0];
        b1 = ((double*)sumX.data)[1];
        c1 = ((double*)sumX.data)[2];
        d1 = ((double*)sumX.data)[3];
        e1 = ((double*)sumX.data)[4];
    }
    else
    {
        a1 = ((float*)sumX.data)[0];
        b1 = ((float*)sumX.data)[1];
        c1 = ((float*)sumX.data)[2];
        d1 = ((float*)sumX.data)[3];
        e1 = ((float*)sumX.data)[4];
    }

    orientation_tolerance = 1.0e-3;

    /* remove the orientation from the ellipse */
    if (MIN(fabs(b1/a1), fabs(b1/c1)) > orientation_tolerance)
    {
        orientation_rad = 0.5 * atan2(b1,(c1-a1));
        cos_phi = cos(orientation_rad);
        sin_phi = sin(orientation_rad);
        a2 = a1*cos_phi*cos_phi - b1*cos_phi*sin_phi + c1*sin_phi*sin_phi;
        b2 = 0;
        c2 = a1*sin_phi*sin_phi + b1*cos_phi*sin_phi + c1*cos_phi*cos_phi;
        d2 = d1*cos_phi - e1*sin_phi;
        e2 = d1*sin_phi + e1*cos_phi;
        mean_x_2 = cos_phi*mean_x_1 - sin_phi*mean_y_1;
        mean_y_2 = sin_phi*mean_x_1 + cos_phi*mean_y_1;
        a1 = a2; b1 = b2; c1 = c2; d1 = d2; e1 = e2;
        mean_x_1 = mean_x_2; mean_y_1 = mean_y_2;
    }
    else
    {
        orientation_rad = M_PI/2.0;
        cos_phi = cos(orientation_rad); /* FIXME unnecessary */
        sin_phi = sin(orientation_rad);
    }

    /* Check if conic equation represents an ellipse */
    test = a1 * c1;
    if (test <= 0.0 )
    {
        printf("test: %f\n", test);
        return OSKAR_ERR_UNKNOWN;
    }
    else
    {
        if (a1 < 0.0)
        {
            a1 = -a1;
            c1 = -c1;
            d1 = -d1;
            e1 = -e1;
        }

        /* final ellipse parameters */
        F  = 1.0 + (d1*d1)/(4.0*a1) + (e1*e1)/(4.0*c1);
        a1 = sqrt(F/a1);
        b1 = sqrt(F/c1);
        *gauss_maj = 2.0 * MAX(a1, b1);
        *gauss_min = 2.0 * MIN(a1, b1);
        *gauss_phi = orientation_rad;
    }

    /* clean up */
    err = oskar_mem_free(&x2);
    err = oskar_mem_free(&y2);
    err = oskar_mem_free(&X);
    err = oskar_mem_free(&XX);
    err = oskar_mem_free(&sumX);
    err = oskar_mem_free(&ipiv);

    return err;
}

#ifdef __cplusplus
}
#endif
