/*
 * Copyright (c) 2012, The University of Oxford
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

#include "test/Test_dierckx.h"
#include "math/oskar_SplineData.h"
#include "math/oskar_SettingsSpline.h"
#include "math/oskar_spline_data_init.h"
#include "math/oskar_dierckx_surfit.h"
#include "math/oskar_dierckx_bispev.h"
#include "math/cudak/oskar_cudak_dierckx_bispev.h"
#include "math/cudak/oskar_cudak_dierckx_bispev_bicubic.h"
#include "utility/oskar_mem_all_headers.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using std::vector;

/**
 * @details
 * Converts the parameter to a C++ string.
 */
#include <sstream>

template <class T>
inline std::string oskar_to_std_string(const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

extern "C"
{
void bispev_(float tx[], int* nx, float ty[], int* ny, float c[],
        int* kx, int* ky, float x[], int* mx, float y[], int* my,
        float z[], float wrk[], int* lwrk, int iwrk[], int* kwrk, int* ier);

void surfit_(int* iopt, int* m, float* x, float* y, float* z, float* w,
        float* xb, float* xe, float* yb, float* ye, int* kx, int* ky, float* s,
        int* nxest, int* nyest, int* nmax, float* eps, int* nx, float* tx,
        int* ny, float* ty, float* c, float* fp, float* wrk1, int* lwrk1,
        float* wrk2, int* lwrk2, int* iwrk, int* kwrk, int* ier);
}

/* Returns the largest absolute (real) value in the array. */
static double oskar_mem_max_abs(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    if (data->type == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (fabsf(p[i]) > r) r = fabsf(p[i]);
        }
    }
    else if (data->type == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (fabs(p[i]) > r) r = fabs(p[i]);
        }
    }
    return r;
}

/* Returns the largest value in the array. */
static double oskar_mem_max(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    if (data->type == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (p[i] > r) r = p[i];
        }
    }
    else if (data->type == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (p[i] > r) r = p[i];
        }
    }
    return r;
}

/* Returns the smallest value in the array. */
static double oskar_mem_min(const oskar_Mem* data, int n)
{
    int i;
    double r = DBL_MAX;
    if (data->type == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (p[i] < r) r = p[i];
        }
    }
    else if (data->type == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)data->data;
        for (i = 0; i < n; ++i)
        {
            if (p[i] < r) r = p[i];
        }
    }
    return r;
}

static int oskar_spline_data_surfit_fortran(oskar_SplineData* spline,
        int num_points, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        oskar_Mem* w, const oskar_SettingsSpline* settings)
{
    int element_size, err, k = 0, maxiter = 1000, type;
    int b1, b2, bx, by, iopt, km, kwrk, lwrk1, lwrk2, ne, nxest, nyest, u, v;
    int sqrt_num_points;
    int *iwrk;
    void *wrk1, *wrk2;
    float x_beg, x_end, y_beg, y_end;

    /* Order of splines - do not change these values. */
    int kx = 3, ky = 3;

    /* Check that parameters are within allowed ranges. */
    if (settings->smoothness_factor_reduction >= 1.0 ||
            settings->smoothness_factor_reduction <= 0.0)
        return OSKAR_ERR_SETTINGS;
    if (settings->average_fractional_error_factor_increase <= 1.0)
        return OSKAR_ERR_SETTINGS;

    /* Get the data type. */
    type = z->type;
    element_size = oskar_mem_element_size(type);
    if ((type != OSKAR_SINGLE) && (type != OSKAR_DOUBLE))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that input data is on the CPU. */
    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU ||
            w->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Get data boundaries. */
    x_beg = (float) oskar_mem_min(x, num_points);
    x_end = (float) oskar_mem_max(x, num_points);
    y_beg = (float) oskar_mem_min(y, num_points);
    y_end = (float) oskar_mem_max(y, num_points);

    /* Initialise and allocate spline data. */
    sqrt_num_points = (int)sqrt(num_points);
    nxest = kx + 1 + sqrt_num_points;
    nyest = ky + 1 + sqrt_num_points;
    u = nxest - kx - 1;
    v = nyest - ky - 1;
    err = oskar_spline_data_init(spline, type, OSKAR_LOCATION_CPU);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_x, nxest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y, nyest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff, u * v);
    if (err) return err;

    /* Set up workspace. */
    km = 1 + ((kx > ky) ? kx : ky);
    ne = (nxest > nyest) ? nxest : nyest;
    bx = kx * v + ky + 1;
    by = ky * u + kx + 1;
    if (bx <= by)
    {
        b1 = bx;
        b2 = b1 + v - ky;
    }
    else
    {
        b1 = by;
        b2 = b1 + u - kx;
    }
    lwrk1 = u * v * (2 + b1 + b2) +
            2 * (u + v + km * (num_points + ne) + ne - kx - ky) + b2 + 1;
    lwrk2 = u * v * (b2 + 1) + b2;
    kwrk = num_points + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1);
    wrk1 = malloc(lwrk1 * element_size);
    wrk2 = malloc(lwrk2 * element_size);
    iwrk = (int*)malloc(kwrk * sizeof(int));
    if (wrk1 == NULL || wrk2 == NULL || iwrk == NULL)
        return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

    if (type == OSKAR_SINGLE)
    {
        /* Set up the surface fitting parameters. */
        float eps, s, user_s, fp = 0.0;
        float *knots_x, *knots_y, *coeff, peak_abs, avg_frac_err_loc;
        int done = 0;
        eps              = (float)settings->eps_float;
        avg_frac_err_loc = (float)settings->average_fractional_error;
        knots_x          = (float*)spline->knots_x.data;
        knots_y          = (float*)spline->knots_y.data;
        coeff            = (float*)spline->coeff.data;
        peak_abs         = oskar_mem_max_abs(z, num_points);
        user_s           = (float)settings->smoothness_factor_override;
        do
        {
            float avg_err, term;
            avg_err = avg_frac_err_loc * peak_abs;
            term = num_points * avg_err * avg_err; /* Termination condition. */
            s = settings->search_for_best_fit ? 2.0 * term : user_s;
            for (k = 0, iopt = 0; k < maxiter; ++k)
            {
                if (k > 0) iopt = 1; /* Set iopt to 1 if not first pass. */
                surfit_(&iopt, &num_points, (float*)x->data,
                        (float*)y->data, (float*)z->data,
                        (float*)w->data, &x_beg, &x_end,
                        &y_beg, &y_end, &kx, &ky, &s, &nxest, &nyest,
                        &ne, &eps, &spline->num_knots_x, knots_x,
                        &spline->num_knots_y, knots_y, coeff, &fp,
                        (float*)wrk1, &lwrk1, (float*)wrk2, &lwrk2, iwrk,
                        &kwrk, &err);
                printf("Iteration %d, s = %.4e, fp = %.4e\n", k, s, fp);

                /* Check for errors. */
                if (err > 0 || err < -2) break;
                else if (err == -2) s = fp;

                /* Check if the fit is good enough. */
                if (!settings->search_for_best_fit || fp < term || s < term)
                    break;

                /* Decrease smoothing factor. */
                s *= settings->smoothness_factor_reduction;
            }

            /* Check for errors. */
            if (err > 0 || err < -2)
            {
                printf("Error (%d) finding spline coefficients.\n", err);
                if (!settings->search_for_best_fit || err == 10)
                {
                    err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                    goto stop;
                }
                avg_frac_err_loc *= settings->average_fractional_error_factor_increase;
                printf("Increasing allowed average fractional error to %.3f.\n",
                        avg_frac_err_loc);
            }
            else
            {
                done = 1;
                err = 0;
                if (err == 5)
                {
                    printf("Cannot add any more knots.\n");
                    avg_frac_err_loc = sqrt(fp / num_points) / peak_abs;
                }
                if (settings->search_for_best_fit)
                {
                    printf("Surface fit to %.3f avg. frac. error "
                            "(s=%.2e, fp=%.2e, k=%d).\n", avg_frac_err_loc,
                            s, fp, k);
                }
                else
                {
                    printf("Surface fit (s=%.2e, fp=%.2e).\n", s, fp);
                }
                printf("Number of knots (x: %d, y: %d)\n",
                        spline->num_knots_x, spline->num_knots_y);
            }
        } while (settings->search_for_best_fit && !done);
    }

    /* Free work arrays. */
stop:
    free(iwrk);
    free(wrk2);
    free(wrk1);

    return err;
}


void Test_dierckx::test_surfit()
{
#if 0
    // Set data dimensions.
    int size_x_in = 20;
    int size_y_in = 10;
    int num_points = size_x_in * size_y_in;

    // Set up the input data.
    vector<float> x_in(num_points), y_in(num_points), z_in(num_points), w(num_points);
    for (int y = 0, i = 0; y < size_y_in; ++y)
    {
        float y1 = y * (2.0 * M_PI) / (size_y_in - 1);
        for (int x = 0; x < size_x_in; ++x, ++i)
        {
            float x1 = x * (M_PI / 2.0) / (size_x_in - 1);

            // Store the data points.
            x_in[i] = x1;
            y_in[i] = y1;
            z_in[i] = cos(x1); // Value of the function at x,y.
            w[i]    = 1.0; // Weight.
        }
    }

    // Set up the surface fitting parameters.
    float noise = 5e-4; // Numerical noise on input data.
    float eps = 1e-6; // Magnitude of float epsilon.

    // Order of splines - do not change these values.
    int kx = 3, ky = 3;

    // Set up workspace.
    int sqrt_num_points = (int)sqrt(num_points);
    int nxest = kx + 1 + sqrt_num_points;
    int nyest = ky + 1 + sqrt_num_points;
    int u = nxest - kx - 1;
    int v = nyest - ky - 1;
    int ncoeff = u * v;
    int km = 1 + ((kx > ky) ? kx : ky);
    int ne = (nxest > nyest) ? nxest : nyest;
    int bx = kx * v + ky + 1;
    int by = ky * u + kx + 1;
    int b1, b2;
    if (bx <= by)
    {
        b1 = bx;
        b2 = b1 + v - ky;
    }
    else
    {
        b1 = by;
        b2 = b1 + u - kx;
    }
    int lwrk1 = u * v * (2 + b1 + b2) +
            2 * (u + v + km * (num_points + ne) + ne - kx - ky) + b2 + 1;
    int lwrk2 = u * v * (b2 + 1) + b2;
    int kwrk = num_points + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1);
    vector<float> wrk1(lwrk1), wrk2(lwrk2);
    vector<int> iwrk(kwrk);

    int k = 0, ier = 0;
    float s;

    // Set up the spline knots (Fortran).
    int nx_f = 0, ny_f = 0; // Number of knots in x and y.
    vector<float> tx_f(nxest, 0.0), ty_f(nyest, 0.0); // Knots in x and y.
    vector<float> c_f(ncoeff, 0.0); // Spline coefficients.
    float fp_f = 0.0; // Sum of squared residuals.
    {
        // Set initial smoothing factor.
        s = num_points + sqrt(2.0 * num_points);
        int iopt = 0;
        TIMER_START
        for (k = 0; k < 1000; ++k)
        {
            if (k > 0) iopt = 1; // Set iopt to 1 if not the first pass.
            surfit_(&iopt, &num_points, &x_in[0],
                    &y_in[0], &z_in[0], &w[0], &x_beg, &x_end,
                    &y_beg, &y_end, &kx, &ky, &s, &nxest, &nyest,
                    &ne, &eps, num_knots_x, knots_x, num_knots_y,
                    knots_y, coeff, &fp, (float*)wrk1, &lwrk1,
                    (float*)wrk2, &lwrk2, iwrk, &kwrk, &err);

            surfit_(&iopt, &num_points, &x_in[0], &y_in[0], &z_in[0], &w[0], &s,
                    &ntest, &npest, &eps, &nx_f, &tx_f[0], &ny_f, &ty_f[0],
                    &c_f[0], &fp_f, &wrk1[0], &lwrk1, &wrk2[0], &lwrk2,
                    &iwrk[0], &kwrk, &ier);

            // Check return code.
            if (ier > 0 || ier < -2)
                CPPUNIT_FAIL("Spline coefficient computation failed with code "
                        + oskar_to_std_string(ier));
            else if (ier == -2) s = fp_f * 0.9;
            else s /= 1.2;

            // Check if the fit is good enough.
            if ((fp_f / num_points) < pow(2.0 * noise, 2)) break;
        }
        TIMER_STOP("Finished surfit precalculation [Fortran]");
    }

    // Set up the spline knots (C).
    int nx_c = 0, ny_c = 0; // Number of knots in x and y.
    vector<float> tx_c(ntest, 0.0), ty_c(npest, 0.0); // Knots in x and y.
    vector<float> c_c(ncoeff, 0.0); // Spline coefficients.
    float fp_c = 0.0; // Sum of squared residuals.
    {
        // Set initial smoothing factor.
        s = num_points + sqrt(2.0 * num_points);
        int iopt = 0;
        TIMER_START
        for (k = 0; k < 1000; ++k)
        {
            if (k > 0) iopt = 1; // Set iopt to 1 if not the first pass.
            oskar_dierckx_surfit_f(iopt, num_points, &x_in[0], &y_in[0], &z_in[0], &w[0], s,
                    ntest, npest, eps, &nx_c, &tx_c[0], &ny_c, &ty_c[0],
                    &c_c[0], &fp_c, &wrk1[0], lwrk1, &wrk2[0], lwrk2,
                    &iwrk[0], kwrk, &ier);

            // Check return code.
            if (ier > 0 || ier < -2)
                CPPUNIT_FAIL("Spline coefficient computation failed with code "
                        + oskar_to_std_string(ier));
            else if (ier == -2) s = fp_c * 0.9;
            else s /= 1.2;

            // Check if the fit is good enough.
            if ((fp_c / num_points) < pow(2.0 * noise, 2)) break;
        }
        TIMER_STOP("Finished surfit precalculation [C]");
    }

    // Check results are consistent.
    double delta = 1e-5;
    CPPUNIT_ASSERT_EQUAL(nx_f, nx_c);
    CPPUNIT_ASSERT_EQUAL(ny_f, ny_c);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(fp_f, fp_c, delta);
    for (int i = 0; i < nx_c; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tx_f[i], tx_c[i], delta);
    for (int i = 0; i < ny_c; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(ty_f[i], ty_c[i], delta);
    for (int i = 0; i < ncoeff; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(c_f[i], c_c[i], delta);

    // Print knot positions.
    printf(" ## Pass %d has knots (nx,ny)=(%d,%d), s=%.6f, fp=%.6f\n",
            k + 1, nx_c, ny_c, s, fp_c);
    printf("    x:\n");
    for (int i = 0; i < nx_c; ++i) printf(" %.3f", tx_c[i]); printf("\n");
    printf("    y:\n");
    for (int i = 0; i < ny_c; ++i) printf(" %.3f", ty_c[i]); printf("\n\n");

    // Output buffers.
    int size_x_out = 100;
    int size_y_out = 200;
    int m_out = size_x_out * size_y_out;
    vector<float> x_out(m_out), y_out(m_out);
    vector<float> z_out_f(m_out), z_out_c(m_out);
    int iwrk1[2];
    float wrk[16];
    int kwrk1 = sizeof(iwrk1) / sizeof(int);
    int lwrk = sizeof(wrk) / sizeof(float);

    // Evaluate output point positions.
    for (int p = 0, i = 0; p < size_y_out; ++p)
    {
        float phi1 = p * (2.0 * M_PI) / (size_y_out - 1);
        for (int t = 0; t < size_x_out; ++t, ++i)
        {
            float theta1 = t * (M_PI / 2.0) / (size_x_out - 1);
            x_out[i] = theta1;
            y_out[i]   = phi1;
        }
    }

    // Evaluate surface (Fortran).
    {
        int one = 1;
        TIMER_START
        for (int i = 0; i < m_out; ++i)
        {
            float val;
            bispev_(&tx_f[0], &nx_f, &ty_f[0], &ny_f, &c_f[0], &kx, &ky,
                    &x_out[i], &one, &y_out[i], &one, &val, wrk, &lwrk,
                    iwrk1, &kwrk1, &ier);
            if (ier != 0)
                CPPUNIT_FAIL("ERROR: Spline evaluation failed\n");
            z_out_f[i] = val;
        }
        TIMER_STOP("Finished surface evaluation [Fortran] (%d points)", m_out);
    }

    // Evaluate surface (C).
    {
        TIMER_START
        for (int i = 0; i < m_out; ++i)
        {
            float val;
            oskar_dierckx_bispev_f(&tx_c[0], nx_c, &ty_c[0], ny_c, &c_c[0], kx, ky,
                    &x_out[i], 1, &y_out[i], 1, &val, wrk, lwrk, iwrk1,
                    kwrk1, &ier);
            if (ier != 0)
                CPPUNIT_FAIL("ERROR: Spline evaluation failed\n");
            z_out_c[i]   = val;
        }
        TIMER_STOP("Finished surface evaluation [C] (%d points)", m_out);
    }

    // Evaluate surface (CUDA).
    oskar_Mem z_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_CPU, m_out);
    {
        int err;

        // Copy memory to GPU.
        oskar_Mem tx_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem ty_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem c_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem x_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem y_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        err = tx_cuda.append_raw(&tx_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, nx_c);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = ty_cuda.append_raw(&ty_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, ny_c);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = c_cuda.append_raw(&c_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, ncoeff);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = x_out_cuda.append_raw(&x_out[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = y_out_cuda.append_raw(&y_out[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Allocate memory for result.
        oskar_Mem z_out_cuda_temp(OSKAR_SINGLE, OSKAR_LOCATION_GPU, m_out);

        // Call kernel.
        int num_blocks, num_threads = 256;
        num_blocks = (m_out + num_threads - 1) / num_threads;
        TIMER_START
        oskar_cudak_dierckx_bispev_bicubic_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (tx_cuda,
                nx_c, ty_cuda, ny_c, c_cuda, m_out, x_out_cuda,
                y_out_cuda, 1, z_out_cuda_temp);
        cudaDeviceSynchronize();
        err = (int) cudaPeekAtLastError();
        TIMER_STOP("Finished sphere evaluation [CUDA] (%d points)", m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Copy memory back.
        err = z_out_cuda_temp.copy_to(&z_out_cuda);
        CPPUNIT_ASSERT_EQUAL(0, err);
    }

    // Check results are consistent.
    for (int i = 0; i < m_out; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(z_out_f[i], z_out_c[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(z_out_c[i], ((float*)z_out_cuda)[i], 1e-6);
    }

    // Write out the data.
    FILE* file = fopen("test_surfit.dat", "w");
    for (int i = 0; i < m_out; ++i)
    {
        fprintf(file, "%10.6f %10.6f %10.6f\n ",
                x_out[i], y_out[i], z_out_c[i]);
    }
    fclose(file);
#endif
}
