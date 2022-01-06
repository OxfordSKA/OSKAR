/*
 * SELECTED LAPACK ROUTINES
 *
 * Copyright (c) 1992-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2000-2013 The University of California Berkeley. All
 *                         rights reserved.
 * Copyright (c) 2006-2013 The University of Colorado Denver.  All rights
 *                         reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer listed
 *   in this license in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties.  The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This C translation of the original FORTRAN source
 * is also covered by the 3-clause BSD License, as follows:
 *
 * Copyright (c) 2016, The University of Oxford
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

#include <math.h>
#include <ctype.h>
#include <string.h>
#include <float.h>

#include "math/oskar_lapack_subset.h"

#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

static void oskar_dgetrf2(const int m, const int n, double *A,
        const int lda, int *ipiv, int *info);
static void oskar_sgetrf2(const int m, const int n, float *A,
        const int lda, int *ipiv, int *info);

static void oskar_dlaswp(const int n, double *A, const int lda,
        const int k1, const int k2, int *ipiv, const int incx);
static void oskar_slaswp(const int n, float *A, const int lda,
        const int k1, const int k2, int *ipiv, const int incx);

static int oskar_ieeeck(const int ispec, const float zero, const float one);
static int oskar_ilaenv(const int ispec, const char *name,
        const int n1, const int n2, const int n3, const int n4);
static int oskar_iparmq(const int ispec, const char *name,
        const int ilo, const int ihi);

/* BLAS functions. */
static int oskar_idamax(const int n, double *dx, const int incx);
static int oskar_isamax(const int n, float *dx, const int incx);
static void oskar_dscal(const int n, const double da, double *dx,
        const int incx);
static void oskar_sscal(const int n, const float da, float *dx,
        const int incx);
static void oskar_dgemm(const char *transa, const char *transb,
        const int m, const int n, const int k, const double alpha,
        const double *A, const int lda, const double *B, const int ldb,
        const double beta, double *C, const int ldc);
static void oskar_sgemm(const char *transa, const char *transb,
        const int m, const int n, const int k, const float alpha,
        const float *A, const int lda, const float *B, const int ldb,
        const float beta, float *C, const int ldc);
static void oskar_dtrsm(const char *side, const char *uplo, const char *transa,
        const char *diag, const int m, const int n, const double alpha,
        const double *A, const int lda, double *B, const int ldb);
static void oskar_strsm(const char *side, const char *uplo, const char *transa,
        const char *diag, const int m, const int n, const float alpha,
        const float *A, const int lda, float *B, const int ldb);

void oskar_dgetrf(const int m, const int n, double *A, const int lda,
        int *ipiv, int *info)
{
    int a_offset = 0, i1 = 0, i2 = 0, i3 = 0;
    int i = 0, j = 0, jb = 0, nb = 0, iinfo = 0, mindim = 0;

    if (m == 0 || n == 0) return;

    /* Determine the block size for this environment. */
    mindim = min(m, n);
    nb = oskar_ilaenv(1, "DGETRF", m, n, -1, -1);
    if (nb <= 1 || nb >= mindim)
    {
        /* Use unblocked code. */
        oskar_dgetrf2(m, n, A, lda, ipiv, info);
    }
    else
    {
        /* Parameter adjustments */
        a_offset = 1 + lda;
        A -= a_offset;
        --ipiv;

        /* Use blocked code. */
        for (j = 1; nb < 0 ? j >= mindim : j <= mindim; j += nb)
        {
            /* Computing MIN */
            i1 = mindim - j + 1;
            jb = min(i1, nb);

            /* Factor diagonal and subdiagonal blocks and test for exact
             * singularity. */
            i1 = m - j + 1;
            oskar_dgetrf2(i1, jb, &A[j + j * lda], lda, &ipiv[j], &iinfo);

            /* Adjust INFO and the pivot indices. */
            if (*info == 0 && iinfo > 0) *info = iinfo + j - 1;

            /* Computing MIN */
            i3 = j + jb - 1;
            i1 = min(m, i3);
            for (i = j; i <= i1; ++i) ipiv[i] += (j - 1);

            /* Apply interchanges to columns 1:J-1. */
            oskar_dlaswp(j - 1, &A[a_offset], lda, j, j + jb - 1, &ipiv[1], 1);

            if (j + jb <= n)
            {
                /* Apply interchanges to columns J+JB:N. */
                i1 = n - j - jb + 1;
                i2 = j + jb - 1;
                oskar_dlaswp(i1, &A[(j + jb) * lda + 1],
                        lda, j, i2, &ipiv[1], 1);

                /* Compute block row of U. */
                i1 = n - j - jb + 1;
                oskar_dtrsm("Left", "Lower", "No transpose", "Unit",
                        jb, i1, 1.0, &A[j + j * lda], lda,
                        &A[j + (j + jb) * lda], lda);
                if (j + jb <= m)
                {
                    /* Update trailing submatrix. */
                    i1 = m - j - jb + 1;
                    i2 = n - j - jb + 1;
                    oskar_dgemm("No transpose", "No transpose", i1, i2, jb,
                            -1.0, &A[j + jb + j * lda], lda,
                            &A[j + (j + jb) * lda], lda, 1.0,
                            &A[j + jb + (j + jb) * lda], lda);
                }
            }
        }
    }
}


void oskar_sgetrf(const int m, const int n, float *A, const int lda,
        int *ipiv, int *info)
{
    int a_offset = 0, i1 = 0, i2 = 0, i3 = 0;
    int i = 0, j = 0, jb = 0, nb = 0, iinfo = 0, mindim = 0;

    if (m == 0 || n == 0) return;

    /* Determine the block size for this environment. */
    mindim = min(m, n);
    nb = oskar_ilaenv(1, "SGETRF", m, n, -1, -1);
    if (nb <= 1 || nb >= mindim)
    {
        /* Use unblocked code. */
        oskar_sgetrf2(m, n, A, lda, ipiv, info);
    }
    else
    {
        /* Parameter adjustments */
        a_offset = 1 + lda;
        A -= a_offset;
        --ipiv;

        /* Use blocked code. */
        for (j = 1; nb < 0 ? j >= mindim : j <= mindim; j += nb)
        {
            /* Computing MIN */
            i1 = mindim - j + 1;
            jb = min(i1, nb);

            /* Factor diagonal and subdiagonal blocks and test for exact
             * singularity. */
            i1 = m - j + 1;
            oskar_sgetrf2(i1, jb, &A[j + j * lda], lda, &ipiv[j], &iinfo);

            /* Adjust INFO and the pivot indices. */
            if (*info == 0 && iinfo > 0) *info = iinfo + j - 1;

            /* Computing MIN */
            i3 = j + jb - 1;
            i1 = min(m, i3);
            for (i = j; i <= i1; ++i) ipiv[i] += (j - 1);

            /* Apply interchanges to columns 1:J-1. */
            oskar_slaswp(j - 1, &A[a_offset], lda, j, j + jb - 1, &ipiv[1], 1);

            if (j + jb <= n)
            {
                /* Apply interchanges to columns J+JB:N. */
                i1 = n - j - jb + 1;
                i2 = j + jb - 1;
                oskar_slaswp(i1, &A[(j + jb) * lda + 1],
                        lda, j, i2, &ipiv[1], 1);

                /* Compute block row of U. */
                i1 = n - j - jb + 1;
                oskar_strsm("Left", "Lower", "No transpose", "Unit", jb, i1,
                        1.0f, &A[j + j * lda], lda,
                        &A[j + (j + jb) * lda], lda);
                if (j + jb <= m)
                {
                    /* Update trailing submatrix. */
                    i1 = m - j - jb + 1;
                    i2 = n - j - jb + 1;
                    oskar_sgemm("No transpose", "No transpose", i1, i2, jb,
                            -1.0f, &A[j + jb + j * lda], lda,
                            &A[j + (j + jb) * lda], lda, 1.0f,
                            &A[j + jb + (j + jb) * lda], lda);
                }
            }
        }
    }
}


void oskar_dgetrf2(const int m, const int n, double *A,
        const int lda, int *ipiv, int *info)
{
    int a_offset = 0, i = 0, n1 = 0, n2 = 0, iinfo = 0, mindim = 0;
    double temp = 0.0;

    /* Parameter adjustments */
    a_offset = 1 + lda;
    A -= a_offset;
    --ipiv;

    if (m == 0 || n == 0) return;
    if (m == 1)
    {
        /* Use unblocked code for one row case.
         * Just need to handle IPIV and INFO. */
        ipiv[1] = 1;
        if (A[a_offset] == 0.) *info = 1;
    }
    else if (n == 1)
    {
        /* Use unblocked code for one column case. */
        /* Find pivot and test for singularity. */
        i = oskar_idamax(m, &A[a_offset], 1);
        ipiv[1] = i;
        if (A[i + lda] != 0.)
        {
            /* Apply the interchange */
            if (i != 1)
            {
                temp = A[a_offset];
                A[a_offset] = A[i + lda];
                A[i + lda] = temp;
            }

            /* Compute elements 2:M of the column */
            if (fabs(A[a_offset]) >= DBL_MIN)
            {
                oskar_dscal(m - 1, 1. / A[a_offset], &A[lda + 2], 1);
            }
            else
            {
                for (i = 1; i <= m - 1; ++i)
                {
                    A[i + 1 + lda] /= A[a_offset];
                }
            }
        }
        else
        {
            *info = 1;
        }
    }
    else
    {
        /* Use recursive code */
        mindim = min(m,n);
        n1 = mindim / 2;
        n2 = n - n1;

        /*        [ A11 ]
         * Factor [ --- ]
         *        [ A21 ] */
        oskar_dgetrf2(m, n1, &A[a_offset], lda, &ipiv[1], &iinfo);
        if (*info == 0 && iinfo > 0) *info = iinfo;

        /*                       [ A12 ]
         * Apply interchanges to [ --- ]
         *                       [ A22 ] */
        oskar_dlaswp(n2, &A[(n1 + 1) * lda + 1], lda, 1, n1, &ipiv[1], 1);

        /* Solve A12 */
        oskar_dtrsm("L", "L", "N", "U", n1, n2, 1.0, &A[a_offset],
                lda, &A[(n1 + 1) * lda + 1], lda);

        /* Update A22 */
        oskar_dgemm("N", "N", m - n1, n2, n1, -1.0, &A[n1 + 1 + lda],
                lda, &A[(n1 + 1) * lda + 1], lda, 1.0,
                &A[n1 + 1 + (n1 + 1) * lda], lda);

        /* Factor A22 */
        oskar_dgetrf2(m - n1, n2, &A[n1 + 1 + (n1 + 1) * lda], lda,
                &ipiv[n1 + 1], &iinfo);

        /* Adjust INFO and the pivot indices */
        if (*info == 0 && iinfo > 0) *info = iinfo + n1;
        for (i = n1 + 1; i <= mindim; ++i) ipiv[i] += n1;

        /* Apply interchanges to A21 */
        oskar_dlaswp(n1, &A[a_offset], lda, n1 + 1, mindim, &ipiv[1], 1);
    }
}


void oskar_sgetrf2(const int m, const int n, float *A,
        const int lda, int *ipiv, int *info)
{
    int a_offset = 0, i = 0, n1 = 0, n2 = 0, iinfo = 0, mindim = 0;
    float temp = 0.0;

    /* Parameter adjustments */
    a_offset = 1 + lda;
    A -= a_offset;
    --ipiv;

    if (m == 0 || n == 0) return;
    if (m == 1)
    {
        /* Use unblocked code for one row case */
        /* Just need to handle IPIV and INFO */
        ipiv[1] = 1;
        if (A[a_offset] == 0.f) *info = 1;
    }
    else if (n == 1)
    {
        /* Use unblocked code for one column case */
        /* Find pivot and test for singularity */
        i = oskar_isamax(m, &A[a_offset], 1);
        ipiv[1] = i;
        if (A[i + lda] != 0.f)
        {
            /* Apply the interchange */
            if (i != 1)
            {
                temp = A[a_offset];
                A[a_offset] = A[i + lda];
                A[i + lda] = temp;
            }

            /* Compute elements 2:M of the column */
            if (fabs(A[a_offset]) >= FLT_MIN)
            {
                oskar_sscal(m - 1, 1.f / A[a_offset], &A[lda + 2], 1);
            }
            else
            {
                for (i = 1; i <= m - 1; ++i)
                {
                    A[i + 1 + lda] /= A[a_offset];
                }
            }
        }
        else
        {
            *info = 1;
        }
    }
    else
    {
        /* Use recursive code */
        mindim = min(m,n);
        n1 = mindim / 2;
        n2 = n - n1;

        /*        [ A11 ]
         * Factor [ --- ]
         *        [ A21 ] */

        oskar_sgetrf2(m, n1, &A[a_offset], lda, &ipiv[1], &iinfo);
        if (*info == 0 && iinfo > 0) *info = iinfo;

        /*                       [ A12 ]
         * Apply interchanges to [ --- ]
         *                       [ A22 ] */
        oskar_slaswp(n2, &A[(n1 + 1) * lda + 1], lda, 1, n1, &ipiv[1], 1);

        /* Solve A12 */
        oskar_strsm("L", "L", "N", "U", n1, n2, 1.0f, &A[a_offset], lda,
                &A[(n1 + 1) * lda + 1], lda);

        /* Update A22 */
        oskar_sgemm("N", "N", m - n1, n2, n1, -1.0f,
                &A[n1 + 1 + lda], lda, &A[(n1 + 1) * lda + 1], lda,
                1.0f, &A[n1 + 1 + (n1 + 1) * lda], lda);

        /* Factor A22 */
        oskar_sgetrf2(m - n1, n2, &A[n1 + 1 + (n1 + 1) * lda], lda,
                &ipiv[n1 + 1], &iinfo);

        /* Adjust INFO and the pivot indices */
        if (*info == 0 && iinfo > 0) *info = iinfo + n1;
        for (i = n1 + 1; i <= mindim; ++i) ipiv[i] += n1;

        /* Apply interchanges to A21 */
        oskar_slaswp(n1, &A[a_offset], lda, n1 + 1, mindim, &ipiv[1], 1);
    }
}


void oskar_dgetrs(const char *trans, const int n, const int nrhs,
        double *A, const int lda, int *ipiv, double *B, const int ldb)
{
    int notran = 0;
    if (n == 0 || nrhs == 0) return;

    notran = !strncmp(trans, "N", 1) || !strncmp(trans, "n", 1);
    if (notran)
    {
        /* Solve A * X = B. */
        /* Apply row interchanges to the right hand sides. */
        oskar_dlaswp(nrhs, B, ldb, 1, n, ipiv, 1);

        /* Solve L*X = B, overwriting B with X. */
        oskar_dtrsm("Left", "Lower", "No transpose", "Unit",
                n, nrhs, 1.0, A, lda, B, ldb);

        /* Solve U*X = B, overwriting B with X. */
        oskar_dtrsm("Left", "Upper", "No transpose", "Non-unit",
                n, nrhs, 1.0, A, lda, B, ldb);
    }
    else
    {
        /* Solve A**T * X = B. */
        /* Solve U**T *X = B, overwriting B with X. */
        oskar_dtrsm("Left", "Upper", "Transpose", "Non-unit",
                n, nrhs, 1.0, A, lda, B, ldb);

        /* Solve L**T *X = B, overwriting B with X. */
        oskar_dtrsm("Left", "Lower", "Transpose", "Unit",
                n, nrhs, 1.0, A, lda, B, ldb);

        /* Apply row interchanges to the solution vectors. */
        oskar_dlaswp(nrhs, B, ldb, 1, n, ipiv, -1);
    }
}


void oskar_sgetrs(const char *trans, const int n, const int nrhs,
        float *A, const int lda, int *ipiv, float *B, const int ldb)
{
    int notran = 0;
    if (n == 0 || nrhs == 0) return;

    notran = !strncmp(trans, "N", 1) || !strncmp(trans, "n", 1);
    if (notran)
    {
        /* Solve A * X = B. */
        /* Apply row interchanges to the right hand sides. */
        oskar_slaswp(nrhs, B, ldb, 1, n, ipiv, 1);

        /* Solve L*X = B, overwriting B with X. */
        oskar_strsm("Left", "Lower", "No transpose", "Unit",
                n, nrhs, 1.0f, A, lda, B, ldb);

        /* Solve U*X = B, overwriting B with X. */
        oskar_strsm("Left", "Upper", "No transpose", "Non-unit",
                n, nrhs, 1.0f, A, lda, B, ldb);
    }
    else
    {
        /* Solve A**T * X = B. */
        /* Solve U**T *X = B, overwriting B with X. */
        oskar_strsm("Left", "Upper", "Transpose", "Non-unit",
                n, nrhs, 1.0f, A, lda, B, ldb);

        /* Solve L**T *X = B, overwriting B with X. */
        oskar_strsm("Left", "Lower", "Transpose", "Unit",
                n, nrhs, 1.0f, A, lda, B, ldb);

        /* Apply row interchanges to the solution vectors. */
        oskar_slaswp(nrhs, B, ldb, 1, n, ipiv, -1);
    }
}


#define LASWP_MACRO \
        int i = 0, j = 0, k = 0, i1 = 0, i2 = 0, n32 = 0;                     \
        int ip = 0, ix = 0, ix0 = 0, inc = 0;                                 \
        A -= (1 + lda);                                                       \
        --ipiv;                                                               \
        if (incx > 0)                                                         \
        {                                                                     \
            ix0 = k1;                                                         \
            i1 = k1;                                                          \
            i2 = k2;                                                          \
            inc = 1;                                                          \
        }                                                                     \
        else if (incx < 0)                                                    \
        {                                                                     \
            ix0 = (1 - k2) * incx + 1;                                        \
            i1 = k2;                                                          \
            i2 = k1;                                                          \
            inc = -1;                                                         \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            return;                                                           \
        }                                                                     \
        n32 = n / 32 << 5;                                                    \
        if (n32 != 0)                                                         \
        {                                                                     \
            for (j = 1; j <= n32; j += 32)                                    \
            {                                                                 \
                ix = ix0;                                                     \
                for (i = i1; inc < 0 ? i >= i2 : i <= i2; i += inc)           \
                {                                                             \
                    ip = ipiv[ix];                                            \
                    if (ip != i)                                              \
                    {                                                         \
                        for (k = j; k <= j + 31; ++k)                         \
                        {                                                     \
                            temp = A[i + k * lda];                            \
                            A[i + k * lda] = A[ip + k * lda];                 \
                            A[ip + k * lda] = temp;                           \
                        }                                                     \
                    }                                                         \
                    ix += incx;                                               \
                }                                                             \
            }                                                                 \
        }                                                                     \
        if (n32 != n)                                                         \
        {                                                                     \
            ++n32;                                                            \
            ix = ix0;                                                         \
            for (i = i1; inc < 0 ? i >= i2 : i <= i2; i += inc)               \
            {                                                                 \
                ip = ipiv[ix];                                                \
                if (ip != i)                                                  \
                {                                                             \
                    for (k = n32; k <= n; ++k)                                \
                    {                                                         \
                        temp = A[i + k * lda];                                \
                        A[i + k * lda] = A[ip + k * lda];                     \
                        A[ip + k * lda] = temp;                               \
                    }                                                         \
                }                                                             \
                ix += incx;                                                   \
            }                                                                 \
        }


void oskar_dlaswp(const int n, double *A, const int lda,
        const int k1, const int k2, int *ipiv, const int incx)
{
    double temp = 0.0;
    LASWP_MACRO
}


void oskar_slaswp(const int n, float *A, const int lda,
        const int k1, const int k2, int *ipiv, const int incx)
{
    float temp = 0.0f;
    LASWP_MACRO
}


int oskar_ieeeck(const int ispec, const float zero, const float one)
{
    float nan1 = 0.0f, nan2 = 0.0f, nan3 = 0.0f;
    float nan4 = 0.0f, nan5 = 0.0f, nan6 = 0.0f;
    float neginf = 0.0f, posinf = 0.0f, negzro = 0.0f, newzro = 0.0f;

    posinf = one / zero;
    if (posinf <= one) return 0;

    neginf = -(one) / zero;
    if (neginf >= zero) return 0;

    negzro = one / (neginf + one);
    if (negzro != zero) return 0;

    neginf = one / negzro;
    if (neginf >= zero) return 0;

    newzro = negzro + zero;
    if (newzro != zero) return 0;

    posinf = one / newzro;
    if (posinf <= one) return 0;

    neginf *= posinf;
    if (neginf >= zero) return 0;

    posinf *= posinf;
    if (posinf <= one) return 0;

    /* Return if we were only asked to check infinity arithmetic. */
    if (ispec == 0) return 1;

    nan1 = posinf + neginf;
    nan2 = posinf / neginf;
    nan3 = posinf / posinf;
    nan4 = posinf * zero;
    nan5 = neginf * negzro;
    nan6 = nan5 * zero;
    if (nan1 == nan1) return 0;
    if (nan2 == nan2) return 0;
    if (nan3 == nan3) return 0;
    if (nan4 == nan4) return 0;
    if (nan5 == nan5) return 0;
    if (nan6 == nan6) return 0;

    return 1;
}


int oskar_ilaenv(const int ispec, const char *name,
        const int n1, const int n2, const int n3, const int n4)
{
    char c2[2], c3[3], c4[2], subnam[6];
    int i = 0, nb = 0, nx = 0, nbmin = 0, cname = 0, sname = 0;

    /* Convert NAME to upper case if the first character is lower case. */
    strncpy(subnam, name, sizeof(subnam));
    if (subnam[0] == tolower(subnam[0]))
    {
        for (i = 0; i < 6 && subnam[i] != 0; ++i)
        {
            subnam[i] = toupper(subnam[i]);
        }
    }

    sname = subnam[0] == 'S' || subnam[0] == 'D';
    cname = subnam[0] == 'C' || subnam[0] == 'Z';
    if (! (cname || sname))
    {
        return 1;
    }
    strncpy(c2, subnam + 1, sizeof(c2));
    strncpy(c3, subnam + 3, sizeof(c3));
    strncpy(c4, c3 + 1, sizeof(c4));

    switch (ispec)
    {
    case 1:
        /* ISPEC = 1:  block size */

        /* In these examples, separate code is provided for setting NB for
         * float and complex.  We assume that NB will take the same value in
         * single or double precision. */
        nb = 1;
        if (!strncmp(c2, "GE", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                nb = 64;
            }
            else if (!strncmp(c3, "QRF", 3) ||
                    !strncmp(c3, "RQF", 3) ||
                    !strncmp(c3, "LQF",  3) ||
                    !strncmp(c3, "QLF", 3))
            {
                nb = 32;
            }
            else if (!strncmp(c3, "HRD", 3))
            {
                nb = 32;
            }
            else if (!strncmp(c3, "BRD", 3))
            {
                nb = 32;
            }
            else if (!strncmp(c3, "TRI", 3))
            {
                nb = 64;
            }
        }
        else if (!strncmp(c2, "PO", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                nb = 64;
            }
        }
        else if (!strncmp(c2, "SY", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                nb = 64;
            }
            else if (sname && !strncmp(c3, "TRD", 3))
            {
                nb = 32;
            }
            else if (sname && !strncmp(c3, "GST", 3))
            {
                nb = 64;
            }
        }
        else if (cname && !strncmp(c2, "HE", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                nb = 64;
            }
            else if (!strncmp(c3, "TRD", 3))
            {
                nb = 32;
            }
            else if (!strncmp(c3, "GST", 3))
            {
                nb = 64;
            }
        }
        else if (sname && !strncmp(c2, "OR", 2))
        {
            if (c3[0] == 'G')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nb = 32;
                }
            }
            else if (c3[0] == 'M')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nb = 32;
                }
            }
        }
        else if (cname && !strncmp(c2, "UN", 2))
        {
            if (c3[0] == 'G')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nb = 32;
                }
            }
            else if (c3[0] == 'M')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nb = 32;
                }
            }
        }
        else if (!strncmp(c2, "GB", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                if (n4 <= 64)
                {
                    nb = 1;
                }
                else
                {
                    nb = 32;
                }
            }
        }
        else if (!strncmp(c2, "PB", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                if (n2 <= 64)
                {
                    nb = 1;
                }
                else
                {
                    nb = 32;
                }
            }
        }
        else if (!strncmp(c2, "TR", 2))
        {
            if (!strncmp(c3, "TRI", 3))
            {
                nb = 64;
            }
            else if (!strncmp(c3, "EVC", 3))
            {
                nb = 64;
            }
        }
        else if (!strncmp(c2, "LA", 2))
        {
            if (!strncmp(c3, "UUM", 3))
            {
                nb = 64;
            }
        }
        else if (sname && !strncmp(c2, "ST", 2))
        {
            if (!strncmp(c3, "EBZ", 3))
            {
                nb = 1;
            }
        }
        else if (!strncmp(c2, "GG", 2))
        {
            nb = 32;
            if (!strncmp(c3, "HD3", 3))
            {
                nb = 32;
            }
        }
        return nb;
    case 2:
        /* ISPEC = 2:  minimum block size */
        nbmin = 2;
        if (!strncmp(c2, "SY", 2))
        {
            if (!strncmp(c3, "TRF", 3))
            {
                nbmin = 8;
            }
        }
        return nbmin;
    case 3:
        /* ISPEC = 3:  crossover point */
        nx = 0;
        if (!strncmp(c2, "GE", 2))
        {
            if (!strncmp(c3, "QRF", 3) ||
                    !strncmp(c3, "RQF", 3) ||
                    !strncmp(c3, "LQF", 3) ||
                    !strncmp(c3, "QLF", 3))
            {
                nx = 128;
            }
            else if (!strncmp(c3, "HRD", 3))
            {
                nx = 128;
            }
            else if (!strncmp(c3, "BRD", 3))
            {
                nx = 128;
            }
        }
        else if (!strncmp(c2, "SY", 2))
        {
            if (sname && !strncmp(c3, "TRD", 3))
            {
                nx = 32;
            }
        }
        else if (cname && !strncmp(c2, "HE", 2))
        {
            if (!strncmp(c3, "TRD", 3))
            {
                nx = 32;
            }
        }
        else if (sname && !strncmp(c2, "OR", 2))
        {
            if (c3[0] == 'G')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nx = 128;
                }
            }
        }
        else if (cname && !strncmp(c2, "UN", 2))
        {
            if (c3[0] == 'G')
            {
                if (!strncmp(c4, "QR", 2) ||
                        !strncmp(c4, "RQ", 2) ||
                        !strncmp(c4, "LQ", 2) ||
                        !strncmp(c4, "QL", 2) ||
                        !strncmp(c4, "HR", 2) ||
                        !strncmp(c4, "TR", 2) ||
                        !strncmp(c4, "BR", 2))
                {
                    nx = 128;
                }
            }
        }
        else if (!strncmp(c2, "GG", 2))
        {
            nx = 128;
        }
        return nx;
    case 4:
        /* ISPEC = 4:  number of shifts (used by xHSEQR) */
        return 6;
    case 5:
        /* ISPEC = 5:  minimum column dimension (not used) */
        return 2;
    case 6:
        /* ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD) */
        return (int) ((float) min(n1, n2) * 1.6f);
    case 7:
        /* ISPEC = 7:  number of processors (not used) */
        return 1;
    case 8:
        /* ISPEC = 8:  crossover point for multishift (used by xHSEQR) */
        return 50;
    case 9:
        /* ISPEC = 9:  maximum size of the subproblems at the bottom of the */
        /*             computation tree in the divide-and-conquer algorithm */
        /*             (used by xGELSD and xGESDD) */
        return 25;
    case 10:
        /* ISPEC = 10: ieee NaN arithmetic can be trusted not to trap */
        return oskar_ieeeck(1, 0.0f, 1.0f);
    case 11:
        /* ISPEC = 11: infinity arithmetic can be trusted not to trap */
        return oskar_ieeeck(0, 0.0f, 1.0f);
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
        /* 12 <= ISPEC <= 16: xHSEQR or related subroutines. */
        return oskar_iparmq(ispec, name, n2, n3);
    }

    /* Invalid value for ISPEC */
    return -1;
}


int oskar_iparmq(const int ispec, const char *name,
        const int ilo, const int ihi)
{
    int ret_val = 0, t = 0, i = 0, nh = 0, ns = 0;
    char subnam[6];

    if (ispec == 15 || ispec == 13 || ispec == 16)
    {
        /* Set the number simultaneous shifts. */
        nh = ihi - ilo + 1;
        ns = 2;
        if (nh >= 30)
        {
            ns = 4;
        }
        if (nh >= 60)
        {
            ns = 10;
        }
        if (nh >= 150)
        {
            /* Computing MAX */
            t = nh / ((int) round(log((float) nh) / log(2.f)));
            ns = max(10, t);
        }
        if (nh >= 590)
        {
            ns = 64;
        }
        if (nh >= 3000)
        {
            ns = 128;
        }
        if (nh >= 6000)
        {
            ns = 256;
        }
        /* Computing MAX */
        t = ns - ns % 2;
        ns = max(2, t);
    }

    switch (ispec)
    {
    case 12:
        /*        ===== Matrices of order smaller than NMIN get sent
         *              to xLAHQR, the classic double shift algorithm.
         *              This must be at least 11. ==== */
        return 75;
    case 13:
        /*        ==== NW: deflation window size.  ==== */
        return nh <= 500 ? ns : ns * 3 / 2;
    case 14:
        /*        ==== INIBL: skip a multi-shift qr iteration and
         *             whenever aggressive early deflation finds
         *             at least (NIBBLE*(window size)/100) deflations. ==== */
        return 14;
    case 15:
        /*        ==== NSHFTS: The number of simultaneous shifts ===== */
        return ns;
    case 16:
        /*        ==== IACC22: Whether to accumulate reflections
         *             before updating the far-from-diagonal elements
         *             and whether to use 2-by-2 block structure while
         *             doing it.  A small amount of work could be saved
         *             by making this choice dependent also upon the
         *             NH=IHI-ILO+1. */

        /* Convert NAME to upper case if the first character is lower case. */
        strncpy(subnam, name, sizeof(subnam));
        if (subnam[0] == tolower(subnam[0]))
        {
            for (i = 0; i < 6 && subnam[i] != 0; ++i)
            {
                subnam[i] = toupper(subnam[i]);
            }
        }

        if (!strncmp(subnam + 1, "GGHRD", 5) ||
                !strncmp(subnam + 1, "GGHD3", 5))
        {
            ret_val = 1;
            if (nh >= 14)
            {
                ret_val = 2;
            }
        }
        else if (!strncmp(subnam + 3, "EXC", 3))
        {
            if (nh >= 14)
            {
                ret_val = 2;
            }
        }
        else if (!strncmp(subnam + 1, "HSEQR", 5) ||
                !strncmp(subnam + 1, "LAQR", 4))
        {
            if (ns >= 14)
            {
                ret_val = 2;
            }
        }
        return ret_val;
    default:
        /*        ===== invalid value of ispec ===== */
        return -1;
    }
}


/* BLAS functions. */

#define IMAX_MACRO \
        int i = 0, ix = 0, ret_val = 1;                                      \
        --dx;                                                                \
        if (n < 1 || incx <= 0) return 0;                                    \
        if (n == 1) return 1;                                                \
        if (incx == 1)                                                       \
        {                                                                    \
            mx = fabs(dx[1]);                                                \
            for (i = 2; i <= n; ++i)                                         \
            {                                                                \
                if (fabs(dx[i]) > mx)                                        \
                {                                                            \
                    ret_val = i;                                             \
                    mx = fabs(dx[i]);                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            ix = 1;                                                          \
            mx = fabs(dx[1]);                                                \
            ix += incx;                                                      \
            for (i = 2; i <= n; ++i)                                         \
            {                                                                \
                if (fabs(dx[ix]) > mx)                                       \
                {                                                            \
                    ret_val = i;                                             \
                    mx = fabs(dx[ix]);                                       \
                }                                                            \
                ix += incx;                                                  \
            }                                                                \
        }                                                                    \
        return ret_val;


int oskar_idamax(const int n, double *dx, const int incx)
{
    double mx = 0.0;
    IMAX_MACRO
}


int oskar_isamax(const int n, float *dx, const int incx)
{
    float mx = 0.0f;
    IMAX_MACRO
}


#define GEMM_MACRO \
        int i = 0, j = 0, l = 0, nota = 0, notb = 0;                         \
        if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))  \
            return;                                                          \
        nota = !strncmp(transa, "N", 1) || !strncmp(transa, "n", 1);         \
        notb = !strncmp(transb, "N", 1) || !strncmp(transb, "n", 1);         \
        if (alpha == zero)                                                   \
        {                                                                    \
            if (beta == zero)                                                \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    for (i = 0; i < m; ++i) C[i + j * ldc] = zero;           \
                }                                                            \
            }                                                                \
            else                                                             \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    for (i = 0; i < m; ++i) C[i + j * ldc] *= beta;          \
                }                                                            \
            }                                                                \
            return;                                                          \
        }                                                                    \
        if (notb)                                                            \
        {                                                                    \
            if (nota)                                                        \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    if (beta == zero)                                        \
                    {                                                        \
                        for (i = 0; i < m; ++i) C[i + j * ldc] = zero;       \
                    }                                                        \
                    else if (beta != one)                                    \
                    {                                                        \
                        for (i = 0; i < m; ++i) C[i + j * ldc] *= beta;      \
                    }                                                        \
                    for (l = 0; l < k; ++l)                                  \
                    {                                                        \
                        x = alpha * B[l + j * ldb];                          \
                        for (i = 0; i < m; ++i)                              \
                        {                                                    \
                            C[i + j * ldc] += x * A[i + l * lda];            \
                        }                                                    \
                    }                                                        \
                }                                                            \
            }                                                                \
            else                                                             \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    for (i = 0; i < m; ++i)                                  \
                    {                                                        \
                        x = zero;                                            \
                        for (l = 0; l < k; ++l)                              \
                        {                                                    \
                            x += A[l + i * lda] * B[l + j * ldb];            \
                        }                                                    \
                        if (beta == zero)                                    \
                        {                                                    \
                            C[i + j * ldc] = alpha * x;                      \
                        }                                                    \
                        else                                                 \
                        {                                                    \
                            C[i + j * ldc] = alpha * x + beta * C[i + j * ldc]; \
                        }                                                    \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            if (nota)                                                        \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    if (beta == zero)                                        \
                    {                                                        \
                        for (i = 0; i < m; ++i) C[i + j * ldc] = zero;       \
                    }                                                        \
                    else if (beta != one)                                    \
                    {                                                        \
                        for (i = 0; i < m; ++i) C[i + j * ldc] *= beta;      \
                    }                                                        \
                    for (l = 0; l < k; ++l)                                  \
                    {                                                        \
                        x = alpha * B[j + l * ldb];                          \
                        for (i = 0; i < m; ++i)                              \
                        {                                                    \
                            C[i + j * ldc] += x * A[i + l * lda];            \
                        }                                                    \
                    }                                                        \
                }                                                            \
            }                                                                \
            else                                                             \
            {                                                                \
                for (j = 0; j < n; ++j)                                      \
                {                                                            \
                    for (i = 0; i < m; ++i)                                  \
                    {                                                        \
                        x = zero;                                            \
                        for (l = 0; l < k; ++l)                              \
                        {                                                    \
                            x += A[l + i * lda] * B[j + l * ldb];            \
                        }                                                    \
                        if (beta == zero)                                    \
                        {                                                    \
                            C[i + j * ldc] = alpha * x;                      \
                        }                                                    \
                        else                                                 \
                        {                                                    \
                            C[i + j * ldc] = alpha * x + beta * C[i + j * ldc]; \
                        }                                                    \
                    }                                                        \
                }                                                            \
            }                                                                \
        }


void oskar_dgemm(const char *transa, const char *transb,
        const int m, const int n, const int k, const double alpha,
        const double *A, const int lda, const double *B, const int ldb,
        const double beta, double *C, const int ldc)
{
    double x = 0.0;
    const double one = 1., zero = 0.;
    GEMM_MACRO
}


void oskar_sgemm(const char *transa, const char *transb,
        const int m, const int n, const int k, const float alpha,
        const float *A, const int lda, const float *B, const int ldb,
        const float beta, float *C, const int ldc)
{
    float x = 0.0;
    const float one = 1.f, zero = 0.f;
    GEMM_MACRO
}


#define TRSM_MACRO \
        int i = 0, j = 0, k = 0, lside = 0, upper = 0, nota = 0, nounit = 0;  \
        A -= (1 + lda);                                                       \
        B -= (1 + ldb);                                                       \
        lside = !strncmp(side, "L", 1) || !strncmp(side, "l", 1);             \
        nounit = !strncmp(diag, "N", 1) || !strncmp(diag, "n", 1);            \
        upper = !strncmp(uplo, "U", 1) || !strncmp(uplo, "u", 1);             \
        nota = !strncmp(transa, "N", 1) || !strncmp(transa, "n", 1);          \
        if (m == 0 || n == 0) return;                                         \
        if (alpha == zero)                                                    \
        {                                                                     \
            for (j = 1; j <= n; ++j)                                          \
            {                                                                 \
                for (i = 1; i <= m; ++i) B[i + j * ldb] = zero;               \
            }                                                                 \
            return;                                                           \
        }                                                                     \
        if (lside)                                                            \
        {                                                                     \
            if (nota)                                                         \
            {                                                                 \
                if (upper)                                                    \
                {                                                             \
                    for (j = 1; j <= n; ++j)                                  \
                    {                                                         \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= alpha; \
                        }                                                     \
                        for (k = m; k >= 1; --k)                              \
                        {                                                     \
                            if (B[k + j * ldb] != zero)                       \
                            {                                                 \
                                if (nounit)                                   \
                                {                                             \
                                    B[k + j * ldb] /= A[k + k * lda];         \
                                }                                             \
                                for (i = 1; i <= k - 1; ++i)                  \
                                {                                             \
                                    B[i + j * ldb] -= B[k + j * ldb] * A[i + k * lda]; \
                                }                                             \
                            }                                                 \
                        }                                                     \
                    }                                                         \
                }                                                             \
                else                                                          \
                {                                                             \
                    for (j = 1; j <= n; ++j)                                  \
                    {                                                         \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= alpha; \
                        }                                                     \
                        for (k = 1; k <= m; ++k)                              \
                        {                                                     \
                            if (B[k + j * ldb] != zero)                       \
                            {                                                 \
                                if (nounit)                                   \
                                {                                             \
                                    B[k + j * ldb] /= A[k + k * lda];         \
                                }                                             \
                                for (i = k + 1; i <= m; ++i)                  \
                                {                                             \
                                    B[i + j * ldb] -= B[k + j * ldb] * A[i + k * lda]; \
                                }                                             \
                            }                                                 \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
            else                                                              \
            {                                                                 \
                if (upper)                                                    \
                {                                                             \
                    for (j = 1; j <= n; ++j)                                  \
                    {                                                         \
                        for (i = 1; i <= m; ++i)                              \
                        {                                                     \
                            x = alpha * B[i + j * ldb];                       \
                            for (k = 1; k <= i - 1; ++k)                      \
                            {                                                 \
                                x -= A[k + i * lda] * B[k + j * ldb];         \
                            }                                                 \
                            if (nounit) x /= A[i + i * lda];                  \
                            B[i + j * ldb] = x;                               \
                        }                                                     \
                    }                                                         \
                }                                                             \
                else                                                          \
                {                                                             \
                    for (j = 1; j <= n; ++j)                                  \
                    {                                                         \
                        for (i = m; i >= 1; --i)                              \
                        {                                                     \
                            x = alpha * B[i + j * ldb];                       \
                            for (k = i + 1; k <= m; ++k)                      \
                            {                                                 \
                                x -= A[k + i * lda] * B[k + j * ldb];         \
                            }                                                 \
                            if (nounit) x /= A[i + i * lda];                  \
                            B[i + j * ldb] = x;                               \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            if (nota)                                                         \
            {                                                                 \
                if (upper)                                                    \
                {                                                             \
                    for (j = 1; j <= n; ++j)                                  \
                    {                                                         \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= alpha; \
                        }                                                     \
                        for (k = 1; k <= j - 1; ++k)                          \
                        {                                                     \
                            if (A[k + j * lda] != zero)                       \
                            {                                                 \
                                for (i = 1; i <= m; ++i)                      \
                                {                                             \
                                    B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb]; \
                                }                                             \
                            }                                                 \
                        }                                                     \
                        if (nounit)                                           \
                        {                                                     \
                            x = one / A[j + j * lda];                         \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= x;     \
                        }                                                     \
                    }                                                         \
                }                                                             \
                else                                                          \
                {                                                             \
                    for (j = n; j >= 1; --j)                                  \
                    {                                                         \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= alpha; \
                        }                                                     \
                        for (k = j + 1; k <= n; ++k)                          \
                        {                                                     \
                            if (A[k + j * lda] != zero)                       \
                            {                                                 \
                                for (i = 1; i <= m; ++i)                      \
                                {                                             \
                                    B[i + j * ldb] -= A[k + j * lda] * B[i + k * ldb]; \
                                }                                             \
                            }                                                 \
                        }                                                     \
                        if (nounit)                                           \
                        {                                                     \
                            x = one / A[j + j * lda];                         \
                            for (i = 1; i <= m; ++i) B[i + j * ldb] *= x;     \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
            else                                                              \
            {                                                                 \
                if (upper)                                                    \
                {                                                             \
                    for (k = n; k >= 1; --k)                                  \
                    {                                                         \
                        if (nounit)                                           \
                        {                                                     \
                            x = one / A[k + k * lda];                         \
                            for (i = 1; i <= m; ++i) B[i + k * ldb] *= x;     \
                        }                                                     \
                        for (j = 1; j <= k - 1; ++j)                          \
                        {                                                     \
                            if (A[j + k * lda] != zero)                       \
                            {                                                 \
                                x = A[j + k * lda];                           \
                                for (i = 1; i <= m; ++i)                      \
                                {                                             \
                                    B[i + j * ldb] -= x * B[i + k * ldb];     \
                                }                                             \
                            }                                                 \
                        }                                                     \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + k * ldb] *= alpha; \
                        }                                                     \
                    }                                                         \
                }                                                             \
                else                                                          \
                {                                                             \
                    for (k = 1; k <= n; ++k)                                  \
                    {                                                         \
                        if (nounit)                                           \
                        {                                                     \
                            x = one / A[k + k * lda];                         \
                            for (i = 1; i <= m; ++i) B[i + k * ldb] *= x;     \
                        }                                                     \
                        for (j = k + 1; j <= n; ++j)                          \
                        {                                                     \
                            if (A[j + k * lda] != zero)                       \
                            {                                                 \
                                x = A[j + k * lda];                           \
                                for (i = 1; i <= m; ++i)                      \
                                {                                             \
                                    B[i + j * ldb] -= x * B[i + k * ldb];     \
                                }                                             \
                            }                                                 \
                        }                                                     \
                        if (alpha != one)                                     \
                        {                                                     \
                            for (i = 1; i <= m; ++i) B[i + k * ldb] *= alpha; \
                        }                                                     \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }


void oskar_dtrsm(const char *side, const char *uplo, const char *transa,
        const char *diag, const int m, const int n, const double alpha,
        const double *A, const int lda, double *B, const int ldb)
{
    double x = 0.0;
    const double one = 1., zero = 0.;
    TRSM_MACRO
}


void oskar_strsm(const char *side, const char *uplo, const char *transa,
        const char *diag, const int m, const int n, const float alpha,
        const float *A, const int lda, float *B, const int ldb)
{
    float x = 0.0f;
    const float one = 1.f, zero = 0.f;
    TRSM_MACRO
}


#define SCAL_MACRO \
        int i = 0, m = 0, mp1 = 0, nincx = 0;                                \
        --dx;                                                                \
        if (n <= 0 || incx <= 0) return;                                     \
        if (incx == 1)                                                       \
        {                                                                    \
            m = n % 5;                                                       \
            if (m != 0)                                                      \
            {                                                                \
                for (i = 1; i <= m; ++i) dx[i] = da * dx[i];                 \
                if (n < 5) return;                                           \
            }                                                                \
            mp1 = m + 1;                                                     \
            for (i = mp1; i <= n; i += 5)                                    \
            {                                                                \
                dx[i] = da * dx[i];                                          \
                dx[i + 1] = da * dx[i + 1];                                  \
                dx[i + 2] = da * dx[i + 2];                                  \
                dx[i + 3] = da * dx[i + 3];                                  \
                dx[i + 4] = da * dx[i + 4];                                  \
            }                                                                \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            nincx = n * incx;                                                \
            for (i = 1; incx < 0 ? i >= nincx : i <= nincx; i += incx)       \
            {                                                                \
                dx[i] = da * dx[i];                                          \
            }                                                                \
        }


void oskar_dscal(const int n, const double da, double *dx, const int incx)
{
    SCAL_MACRO
}


void oskar_sscal(const int n, const float da, float *dx, const int incx)
{
    SCAL_MACRO
}
