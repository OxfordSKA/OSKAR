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
#include "math/dierckx_surfit.h"
#include "math/dierckx_bispev.h"
#include "math/cudak/oskar_cudak_dierckx_bispev.h"
#include "math/cudak/oskar_cudak_dierckx_bispev_bicubic.h"
#include "utility/oskar_Mem.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

using std::vector;

void Test_dierckx::test_surfit()
{
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
            dierckx_surfit_f(iopt, num_points, &x_in[0], &y_in[0], &z_in[0], &w[0], s,
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
            dierckx_bispev_f(&tx_c[0], nx_c, &ty_c[0], ny_c, &c_c[0], kx, ky,
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
}
