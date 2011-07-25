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

#include "interferometry/oskar_interferometry_cuda_correlator.h"

#include "math/cudak/oskar_math_cudak_dftw_3d_seq_out.h"
#include "math/cudak/oskar_math_cudak_mat_mul_cc.h"
#include "interferometry/cudak/oskar_interferometry_cudak_correlator.h"
#include "interferometry/cudak/oskar_interferometry_cudak_xyz2uvw.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Single precision.

int oskar_interferometry_cudaf_correlator(int na, const float* ax,
        const float* ay, const float* az, int ns, const float* l,
        const float* m, const float* n, const float* eb, float ra0,
        float dec0, float lst0, int nsdt, float sdt,
        float lambda_bandwidth, float* vis, float* work)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    double tOffset = (double)sdt / 2.0;
    const float2* emat = (float2*)eb;
    float2* kmat = (float2*)work;
    float2* visd = (float2*)vis;
    float* u = (float*)(kmat + (ns * na));
    float* v = u + na;
    float* w = v + na;

    // Set up thread blocks.
    dim3 kThd(64, 4); // Sources, antennas.
    dim3 kBlk((ns + kThd.x - 1) / kThd.x, (na + kThd.y - 1) / kThd.y);
    size_t sMem = (kThd.x + kThd.y) * sizeof(float3);
    dim3 mThd(64, 4); // Sources, antennas.
    dim3 mBlk((ns + mThd.x - 1) / mThd.x, (na + mThd.y - 1) / mThd.y);
    dim3 vThd(256, 1); // Antennas, antennas.
    dim3 vBlk(na, na);
    size_t vsMem = vThd.x * sizeof(float2);
    dim3 rThd(256, 1); // Antennas.
    dim3 rBlk((na + rThd.x - 1) / rThd.x, 1);

    // Loop over integrations.
    for (int i = 0; i < nsdt; ++i)
    {
        // Compute the current LST and hour angle of the phase centre.
        double tInc = i * sdt + tOffset;
        double lst = lst0 + 2 * M_PI * tInc / 86400.0; // Must be double.
        double ha0 = lst - ra0; // Must be double.

        // Compute the station u,v,w coordinates.
        oskar_interferometry_cudakf_xyz2uvw <<<rThd, rBlk>>>
                (na, ax, ay, az, ha0, dec0, u, v, w);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Compute K-matrix of 3D DFT weights.
        oskar_math_cudakf_dftw_3d_seq_out <<<kBlk, kThd, sMem>>>
                (na, u, v, w, ns, l, m, n, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Perform complex matrix element multiply of K with E * B.
        oskar_math_cudakf_mat_mul_cc <<<mBlk, mThd>>>
                (ns, na, kmat, emat, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Call the correlator kernel.
        oskar_interferometry_cudakf_correlator <<<vBlk, vThd, vsMem>>>
                (ns, na, kmat, u, v, l, m, lambda_bandwidth, visd);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;
    }

    return 0;
}

// Double precision.

int oskar_interferometry_cudad_correlator(int na, const double* ax,
        const double* ay, const double* az, int ns, const double* l,
        const double* m, const double* n, const double* eb, double ra0,
        double dec0, double lst0, int nsdt, double sdt,
        double lambda_bandwidth, double* vis, double* work)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    double tOffset = (double)sdt / 2.0;
    const double2* emat = (double2*)eb;
    double2* kmat = (double2*)work;
    double2* visd = (double2*)vis;
    double* u = (double*)(kmat + (ns * na));
    double* v = u + na;
    double* w = v + na;

    // Set up thread blocks.
    dim3 kThd(64, 4); // Sources, antennas.
    dim3 kBlk((ns + kThd.x - 1) / kThd.x, (na + kThd.y - 1) / kThd.y);
    size_t sMem = (kThd.x + kThd.y) * sizeof(double3);
    dim3 mThd(64, 4); // Sources, antennas.
    dim3 mBlk((ns + mThd.x - 1) / mThd.x, (na + mThd.y - 1) / mThd.y);
    dim3 vThd(256, 1); // Antennas, antennas.
    dim3 vBlk(na, na);
    size_t vsMem = vThd.x * sizeof(double2);
    dim3 rThd(256, 1); // Antennas.
    dim3 rBlk((na + rThd.x - 1) / rThd.x, 1);

    // Loop over integrations.
    for (int i = 0; i < nsdt; ++i)
    {
        // Compute the current LST and hour angle of the phase centre.
        double tInc = i * sdt + tOffset;
        double lst = lst0 + 2 * M_PI * tInc / 86400.0; // Must be double.
        double ha0 = lst - ra0; // Must be double.

        // Compute the station u,v,w coordinates.
        oskar_interferometry_cudakd_xyz2uvw <<<rThd, rBlk>>>
                (na, ax, ay, az, ha0, dec0, u, v, w);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Compute K-matrix of 3D DFT weights.
        oskar_math_cudakd_dftw_3d_seq_out <<<kBlk, kThd, sMem>>>
                (na, u, v, w, ns, l, m, n, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Perform complex matrix element multiply of K with E * B.
        oskar_math_cudakd_mat_mul_cc <<<mBlk, mThd>>>
                (ns, na, kmat, emat, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;

        // Call the correlator kernel.
        oskar_interferometry_cudakd_correlator <<<vBlk, vThd, vsMem>>>
                (ns, na, kmat, u, v, l, m, lambda_bandwidth, visd);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) return errCuda;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
