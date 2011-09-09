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


#include "apps/oskar_imager_dft.cu.h"
#include "math/oskar_cuda_dft_c2r_2d.h"
#include "utility/oskar_vector_types.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_imager_dft_d(const unsigned num_vis, const double2* vis, double* u,
        double* v, const double frequency, const unsigned image_size,
        const double* l, double* image)
{
    const double c_0 = 299792458;
    const double wavelength = c_0 / frequency;
    const double wavenumber = 2.0 * M_PI / wavelength;

    // Convert baselines to wavenumber units. -- TODO do this on the GPU.
    for (unsigned i = 0; i < num_vis; ++i)
    {
        u[i] *= wavenumber;
        v[i] *= wavenumber;
    }

    // Setup device memory.
    size_t mem_size_coords = num_vis * sizeof(double);
    double* d_u = NULL;
    cudaMalloc((void**)&d_u, mem_size_coords);
    cudaMemcpy(d_u, u, mem_size_coords, cudaMemcpyHostToDevice);

    double* d_v = NULL;
    cudaMalloc((void**)&d_v, mem_size_coords);
    cudaMemcpy(d_v, v, mem_size_coords, cudaMemcpyHostToDevice);

    double2* d_vis = NULL;
    size_t mem_size_vis = num_vis * sizeof(double2);
    cudaMalloc((void**)&d_vis, mem_size_vis);
    cudaMemcpy(d_vis, vis, mem_size_vis, cudaMemcpyHostToDevice);

    unsigned num_pixels = image_size * image_size;
    size_t mem_size_image = num_pixels * sizeof(double);

    // l, m positions of pixels (equivalent to MATLAB meshgrid)
    double* l_2d = (double*) malloc(mem_size_image);
    double* m_2d = (double*) malloc(mem_size_image);
    for (unsigned j = 0; j < image_size; ++j)
    {
        for (unsigned i = 0; i < image_size; ++i)
        {
            l_2d[j * image_size + i] = l[i];
            m_2d[i * image_size + j] = l[image_size-1-i];
        }
    }

    double* d_image = NULL;
    cudaMalloc((void**)&d_image, mem_size_image);

    double* d_l = NULL;
    cudaMalloc((void**)&d_l, mem_size_image);
    cudaMemcpy(d_l, l_2d, mem_size_image, cudaMemcpyHostToDevice);

    double* d_m = NULL;
    cudaMalloc((void**)&d_m, mem_size_image);
    cudaMemcpy(d_m, m_2d, mem_size_image, cudaMemcpyHostToDevice);

    // Call dft.
    int err = oskar_cuda_dft_c2r_2d_d(num_vis, d_u, d_v, (double*)d_vis,
            num_pixels, d_l, d_m, d_image);

    // Copy back image to host memory.
    cudaMemcpy(image, d_image, mem_size_image, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < num_pixels; ++i)
    {
        image[i] /= (double)num_vis;
    }

    cudaFree(d_image);
    cudaFree(d_l);
    cudaFree(d_m);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_vis);

    free(l_2d);
    free(m_2d);

    return err;
}

#ifdef __cplusplus
}
#endif
