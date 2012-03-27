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

#include "extern/dierckx/test/Test_dierckx.h"
#include "extern/dierckx/sphere.h"
#include "extern/dierckx/bispev.h"
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

void regrid_(int* iopt, int* mx, float x[], int* my, float y[], float z[],
        float* xb, float* xe, float* yb, float* ye, int* kx, int* ky,
        float* s, int* nxest, int* nyest, int* nx, float tx[], int* ny,
        float ty[], float c[], float* fp, float wrk[], int* lwrk, int iwrk[],
        int* kwrk, int* ier);

void sphere_(int* iopt, int* m, float* theta, float* phi,
        float* r, float* w, float* s, int* ntest, int* npest,
        float* eps, int* nt, float* tt, int* np, float* tp, float* c,
        float* fp, float* wrk1, int* lwrk1, float* wrk2, int* lwrk2,
        int* iwrk, int* kwrk, int* ier);
}

using std::vector;

void Test_dierckx::test_regrid()
{
    // Set data dimensions.
    int size_x = 10;
    int size_y = 10;

    // Set the input data.
    vector<float> data(size_x * size_y);

    // Create the data axes.
    vector<float> x(size_x), y(size_y);
    for (int i = 0; i < size_x; ++i)
        x[i] = (float) i;
    for (int i = 0; i < size_y; ++i)
        y[i] = (float) i;

    // Set up the surface fitting parameters.
    int iopt = 0; // -1 = Specify least-squares spline.
    int kxy = 3; // Degree of spline (cubic).
    float noise = 5e-4; // Numerical noise on input data.

    // Checks.
    if (size_x <= kxy || size_y <= kxy)
    {
        CPPUNIT_FAIL("ERROR: Input grid dimensions too small. Aborting.");
        return;
    }

    // Set up the spline knots.
    int nx = 0; // Number of knots in x.
    int ny = 0; // Number of knots in y.
    int nxest = size_x + kxy + 1; // Maximum number of knots in x.
    int nyest = size_y + kxy + 1; // Maximum number of knots in y.
    vector<float> tx(nxest, 0.0); // Spline knots in x.
    vector<float> ty(nyest, 0.0); // Spline knots in y.
    vector<float> c((nxest-kxy-1)*(nyest-kxy-1)); // Output spline coefficients.
    float fp = 0.0; // Output sum of squared residuals of spline approximation.

    // Set up workspace.
    int u = size_y > nxest ? size_y : nxest;
    int lwrk = 4 + nxest * (size_y + 2 * kxy + 5) +
            nyest * (2 * kxy + 5) + size_x * (kxy + 1) +
            size_y * (kxy + 1) + u;
    vector<float> wrk(lwrk);
    int kwrk = 3 + size_x + size_y + nxest + nyest;
    vector<int> iwrk(kwrk);
    int ier = 0; // Output return code.

    TIMER_START
    int k = 0;
    // Set initial smoothing factor (ignored for iopt < 0).
    float s = size_x * size_y * pow(noise, 2.0);
    bool fail = false;
    do
    {
        // Generate knot positions *at grid points* if required.
        if (iopt < 0 || fail)
        {
            int i, k, stride = 1;
            for (k = 0, i = kxy - 1; i <= size_x - kxy + stride; i += stride, ++k)
                tx[k + kxy + 1] = x[i]; // Knot x positions.
            nx = k + 2 * kxy + 1;
            for (k = 0, i = kxy - 1; i <= size_y - kxy + stride; i += stride, ++k)
                ty[k + kxy + 1] = y[i]; // Knot y positions.
            ny = k + 2 * kxy + 1;
        }

        // Set iopt to 1 if this is at least the second of multiple passes.
        if (k > 0) iopt = 1;
        regrid_(&iopt, &size_x, &x[0], &size_y, &y[0], &data[0],
                &x[0], &x[size_x-1], &y[0], &y[size_y-1], &kxy, &kxy, &s,
                &nxest, &nyest, &nx, &tx[0], &ny, &ty[0], &c[0], &fp,
                &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
        if (ier == 1)
        {
            CPPUNIT_FAIL("ERROR: Workspace overflow.");
            return;
        }
        else if (ier == 2)
        {
            fprintf(stderr, "ERROR: Impossible result! (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 3)
        {
            fprintf(stderr, "ERROR: Iteration limit. (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 10)
        {
            CPPUNIT_FAIL("ERROR: Invalid input arguments.");
            return;
        }
        fail = false;

        // Print knot positions.
        printf(" ## Pass %d has knots (nx,ny)=(%d,%d), s=%.4f, fp=%.4f\n",
                k+1, nx, ny, s, fp);
        printf("    x:\n");
        for (int j = 0; j < nx; ++j) printf(" %.3f", tx[j]); printf("\n");
        printf("    y:\n");
        for (int j = 0; j < ny; ++j) printf(" %.3f", ty[j]); printf("\n\n");

        // Reduce smoothness parameter.
        s = s / 1.2;

        // Increment counter.
        ++k;
    } while (fp / ((nx-2*(kxy+1)) * (ny-2*(kxy+1))) > pow(2.0 * noise, 2) &&
            k < 1000 && (iopt >= 0 || fail));
    TIMER_STOP("Finished regular grid precalculation");

    // Evaluate surface.
    int out_x = 701;
    int out_y = 501;
    vector<float> output(out_x * out_y);
    TIMER_START
    int one = 1;
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        float py = y[size_y - 1] * float(j) / (out_y - 1);
        for (int i = 0; i < out_x; ++i, ++k)
        {
            float val;
            float px = x[size_x - 1] * float(i) / (out_x - 1);
            bispev_(&tx[0], &nx, &ty[0], &ny, &c[0], &kxy, &kxy,
                    &px, &one, &py, &one, &val,
                    &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
            if (ier != 0)
            {
                CPPUNIT_FAIL("ERROR: Spline evaluation failed\n");
                return;
            }
            output[k] = val;
        }
    }
    TIMER_STOP("Finished evaluation (%d points)", out_x * out_y);

    // Write out the data.
    FILE* file = fopen("test_regrid.dat", "w");
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        for (int i = 0; i < out_x; ++i, ++k)
        {
            fprintf(file, "%10.6f ", output[k]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void Test_dierckx::test_sphere()
{
    // Set data dimensions.
    int size_theta_in = 20;
    int size_phi_in = 10;
    int m_in = size_theta_in * size_phi_in;

    // Set up the input data.
    vector<float> theta_in(m_in), phi_in(m_in), r_in(m_in), w(m_in);
    for (int p = 0, i = 0; p < size_phi_in; ++p)
    {
        float phi1 = p * (2.0 * M_PI) / (size_phi_in - 1); // Phi.
        for (int t = 0; t < size_theta_in; ++t, ++i)
        {
            float theta1 = t * (M_PI / 2.0) / (size_theta_in - 1); // Theta.

            // Store the data points.
            theta_in[i] = theta1;
            phi_in[i]   = phi1;
            r_in[i]     = cos(theta1); // Value of the function at theta,phi.
            w[i]        = 1.0; // Weight.
        }
    }

    // Set up the surface fitting parameters.
    float noise = 5e-4; // Numerical noise on input data.
    float eps = 1e-6; // Magnitude of float epsilon.

    // Set up workspace.
    int ntest = 8 + (int)sqrt(m_in);
    int npest = ntest;
    int u = ntest - 7;
    int v = npest - 7;
    int lwrk1 = 185 + 52*v + 10*u + 14*u*v + 8*(u-1)*v*v + 8*m_in;
    int lwrk2 = 48 + 21*v + 7*u*v + 4*(u-1)*v*v;
    vector<float> wrk1(lwrk1), wrk2(lwrk2);
    int kwrk = m_in + (ntest - 7) * (npest - 7);
    vector<int> iwrk(kwrk);
    int k = 0, ier = 0;
    float s;

    // Set up the spline knots (Fortran).
    int nt_f = 0, np_f = 0; // Number of knots in theta and phi.
    vector<float> tt_f(ntest, 0.0), tp_f(npest, 0.0); // Knots in theta and phi.
    vector<float> c_f((ntest-4) * (npest-4), 0.0); // Spline coefficients.
    float fp_f = 0.0; // Sum of squared residuals.
    {
        // Set initial smoothing factor.
        s = m_in + sqrt(2.0 * m_in);
        int iopt = 0;
        TIMER_START
        for (k = 0; k < 1000; ++k)
        {
            if (k > 0) iopt = 1; // Set iopt to 1 if not the first pass.
            sphere_(&iopt, &m_in, &theta_in[0], &phi_in[0], &r_in[0], &w[0], &s,
                    &ntest, &npest, &eps, &nt_f, &tt_f[0], &np_f, &tp_f[0],
                    &c_f[0], &fp_f, &wrk1[0], &lwrk1, &wrk2[0], &lwrk2,
                    &iwrk[0], &kwrk, &ier);

            // Check return code.
            if (ier > 0 || ier < -2)
                CPPUNIT_FAIL("Spline coefficient computation failed with code "
                        + oskar_to_std_string(ier));
            else if (ier == -2) s = fp_f * 0.9;
            else s /= 1.2;

            // Check if the fit is good enough.
            if ((fp_f / m_in) < pow(2.0 * noise, 2)) break;
        }
        TIMER_STOP("Finished sphere precalculation [Fortran]");
    }

    // Set up the spline knots (C).
    int nt_c = 0, np_c = 0; // Number of knots in theta and phi.
    vector<float> tt_c(ntest, 0.0), tp_c(npest, 0.0); // Knots in theta and phi.
    vector<float> c_c((ntest-4) * (npest-4), 0.0); // Spline coefficients.
    float fp_c = 0.0; // Sum of squared residuals.
    {
        // Set initial smoothing factor.
        s = m_in + sqrt(2.0 * m_in);
        int iopt = 0;
        TIMER_START
        for (k = 0; k < 1000; ++k)
        {
            if (k > 0) iopt = 1; // Set iopt to 1 if not the first pass.
            sphere_f(iopt, m_in, &theta_in[0], &phi_in[0], &r_in[0], &w[0], s,
                    ntest, npest, eps, &nt_c, &tt_c[0], &np_c, &tp_c[0],
                    &c_c[0], &fp_c, &wrk1[0], lwrk1, &wrk2[0], lwrk2,
                    &iwrk[0], kwrk, &ier);

            // Check return code.
            if (ier > 0 || ier < -2)
                CPPUNIT_FAIL("Spline coefficient computation failed with code "
                        + oskar_to_std_string(ier));
            else if (ier == -2) s = fp_c * 0.9;
            else s /= 1.2;

            // Check if the fit is good enough.
            if ((fp_c / m_in) < pow(2.0 * noise, 2)) break;
        }
        TIMER_STOP("Finished sphere precalculation [C]");
    }

    // Check results are consistent.
    double delta = 1e-5;
    CPPUNIT_ASSERT_EQUAL(nt_f, nt_c);
    CPPUNIT_ASSERT_EQUAL(np_f, np_c);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(fp_f, fp_c, delta);
    for (int i = 0; i < nt_c; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tt_f[i], tt_c[i], delta);
    for (int i = 0; i < np_c; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(tp_f[i], tp_c[i], delta);
    for (int i = 0; i < (ntest-4) * (npest-4); ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL(c_f[i], c_c[i], delta);

    // Print knot positions.
    printf(" ## Pass %d has knots (nt,np)=(%d,%d), s=%.6f, fp=%.6f\n",
            k + 1, nt_c, np_c, s, fp_c);
    printf("    theta:\n");
    for (int i = 0; i < nt_c; ++i) printf(" %.3f", tt_c[i]); printf("\n");
    printf("    phi:\n");
    for (int i = 0; i < np_c; ++i) printf(" %.3f", tp_c[i]); printf("\n\n");

    // Output buffers.
    int size_theta_out = 100;
    int size_phi_out = 200;
    int m_out = size_theta_out * size_phi_out;
    vector<float> theta_out(m_out), phi_out(m_out);
    vector<float> r_out_f(m_out), r_out_c(m_out);
    int iwrk1[2];
    float wrk[16];
    int kwrk1 = sizeof(iwrk1) / sizeof(int);
    int lwrk = sizeof(wrk) / sizeof(float);

    // Evaluate output point positions.
    for (int p = 0, i = 0; p < size_phi_out; ++p)
    {
        float phi1 = p * (2.0 * M_PI) / (size_phi_out - 1);
        for (int t = 0; t < size_theta_out; ++t, ++i)
        {
            float theta1 = t * (M_PI / 2.0) / (size_theta_out - 1);
            theta_out[i] = theta1;
            phi_out[i]   = phi1;
        }
    }

    // Evaluate surface (Fortran).
    {
        int kxy = 3; // Degree of spline (cubic).
        int one = 1;
        TIMER_START
        for (int i = 0; i < m_out; ++i)
        {
            float val;
            bispev_(&tt_f[0], &nt_f, &tp_f[0], &np_f, &c_f[0], &kxy, &kxy,
                    &theta_out[i], &one, &phi_out[i], &one, &val, wrk, &lwrk,
                    iwrk1, &kwrk1, &ier);
            if (ier != 0)
                CPPUNIT_FAIL("ERROR: Spherical spline evaluation failed\n");
            r_out_f[i] = val;
        }
        TIMER_STOP("Finished sphere evaluation [Fortran] (%d points)", m_out);
    }

    // Evaluate surface (C).
    {
        TIMER_START
        for (int i = 0; i < m_out; ++i)
        {
            float val;
            bispev_f(&tt_c[0], nt_c, &tp_c[0], np_c, &c_c[0], 3, 3,
                    &theta_out[i], 1, &phi_out[i], 1, &val, wrk, lwrk, iwrk1,
                    kwrk1, &ier);
            if (ier != 0)
                CPPUNIT_FAIL("ERROR: Spherical spline evaluation failed\n");
            r_out_c[i]   = val;
        }
        TIMER_STOP("Finished sphere evaluation [C] (%d points)", m_out);
    }

    // Evaluate surface (CUDA).
    oskar_Mem r_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_CPU, m_out);
    {
        int err;

        // Copy memory to GPU.
        oskar_Mem tt_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem tp_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem c_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem theta_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        oskar_Mem phi_out_cuda(OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        err = tt_cuda.append_raw(&tt_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = tp_cuda.append_raw(&tp_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = c_cuda.append_raw(&c_c[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = theta_out_cuda.append_raw(&theta_out[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = phi_out_cuda.append_raw(&phi_out[0], OSKAR_SINGLE,
                OSKAR_LOCATION_CPU, m_out);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Allocate memory for result.
        oskar_Mem r_out_cuda_temp(OSKAR_SINGLE, OSKAR_LOCATION_GPU, m_out);

        // Call kernel.
        int num_blocks, num_threads = 256;
        num_blocks = (m_out + num_threads - 1) / num_threads;
        TIMER_START
        oskar_cudak_dierckx_bispev_bicubic_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (tt_cuda,
                nt_c, tp_cuda, np_c, c_cuda, m_out, theta_out_cuda,
                phi_out_cuda, 1, r_out_cuda_temp);
//        oskar_cudak_dierckx_bispev_f
//        OSKAR_CUDAK_CONF(num_blocks, num_threads) (tt_cuda,
//                nt_c, tp_cuda, np_c, c_cuda, 3, 3, m_out, theta_out_cuda,
//                phi_out_cuda, 1, r_out_cuda_temp);
        cudaDeviceSynchronize();
        err = (int) cudaPeekAtLastError();
        CPPUNIT_ASSERT_EQUAL(0, err);
        TIMER_STOP("Finished sphere evaluation [CUDA] (%d points)", m_out);

        // Copy memory back.
        err = r_out_cuda_temp.copy_to(&r_out_cuda);
        CPPUNIT_ASSERT_EQUAL(0, err);
    }

    // Check results are consistent.
    for (int i = 0; i < m_out; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(r_out_f[i], r_out_c[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(r_out_c[i], ((float*)r_out_cuda)[i], 1e-6);
    }

    // Write out the data.
    FILE* file = fopen("test_sphere.dat", "w");
    for (int i = 0; i < m_out; ++i)
    {
        fprintf(file, "%10.6f %10.6f %10.6f\n ",
                theta_out[i], phi_out[i], r_out_c[i]);
    }
    fclose(file);
}
