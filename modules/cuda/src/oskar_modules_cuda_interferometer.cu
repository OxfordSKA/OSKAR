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

#include "modules/cuda/oskar_modules_cuda_interferometer.h"
#include "modules/cuda/oskar_modules_cuda_correlator_bw.h"

#include "cuda/kernels/oskar_cudak_dftw_3d_seq_out.h"
#include "cuda/kernels/oskar_cudak_mat_mul_cc.h"
#include "cuda/kernels/oskar_cudak_correlator.h"
#include "cuda/kernels/oskar_cudak_xyz2uvw.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SEC_PER_DAY 86400.0

// Single precision.

int oskar_modules_cudaf_interferometer1(int na, const float* ax,
        const float* ay, const float* az, int ns, const float* ra,
        const float* dec, const float* brightness, const int* nas,
        const float* asx, const float* asy, float ra0, float dec0,
        float t_vis_start, float dt_vis_ave, int num_vis_ave, int nsdt,
        float sdt, float lambda_bandwidth, float* u, float* v, float* w,
        float* vis, float* ework, float* swork, float* cwork)
{
    // Initialise.
    const double dt_vis_ave_offset = dt_vis_ave / 2.0;

    // Clear the visibilities.
    cudaMemset(vis, 0, sizeof(float2) * na * (na - 1) / 2);

    // Loop over full visibility averages.
    for (int j = 0; j < num_vis_ave; ++j)
    {
        // Time step value.
        double t_ave_start = t_vis_start + j * dt_vis_ave;
        double t_ave = t_ave_start + dt_vis_ave_offset;

        // Compute the local sidereal time in radians.
        double lst = 2 * M_PI * t_ave / SEC_PER_DAY;

        // Determine which sources are above the horizon.

        // Compute hour angle, azimuth and elevation of phase tracking centre.

        // Compute E-Jones (station beam) for each visible source and station
        // geometry.

        // Determine direction cosines of visible sources relative to
        // phase centre.

        // Evaluate and average visibility fringes for a fixed E-Jones.
        oskar_modules_cudaf_correlator_bw(na, ax, ay, az, ns, l, m, n,
                eb, ra0, dec0, lst0, nsdt, sdt, lambda_bandwidth, vis, cwork);
    }

    // Divide visibilities by loop counter.

    // Generate u,v,w coordinates for visibility dump.

    return 0;
}

// Double precision.

int oskar_modules_cudad_interferometer1(int na, const double* ax,
        const double* ay, const double* az, int ns, const double* l,
        const double* m, const double* n, const double* eb, double ra0,
        double dec0, double lst0, int nsdt, double sdt,
        double lambda_bandwidth, double* vis, double* work)
{
    return 0;
}

#ifdef __cplusplus
}
#endif
