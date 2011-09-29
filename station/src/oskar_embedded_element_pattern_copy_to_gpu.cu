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

#include "station/oskar_embedded_element_pattern_copy_to_gpu.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_embedded_element_pattern_copy_to_gpu(
        const oskar_EmbeddedElementPattern* h_data,
        oskar_EmbeddedElementPattern* hd_data)
{
    // Copy the meta-data into the new structure.
    hd_data->n_points = h_data->n_points;
    hd_data->n_phi = h_data->n_phi;
    hd_data->n_theta = h_data->n_theta;
    hd_data->inc_phi = h_data->inc_phi;
    hd_data->inc_theta = h_data->inc_theta;
    hd_data->max_phi = h_data->max_phi;
    hd_data->max_theta = h_data->max_theta;
    hd_data->min_phi = h_data->min_phi;
    hd_data->min_theta = h_data->min_theta;

    // Allocate GPU texture memory to hold the look-up tables.
    cudaMallocPitch((void**)&hd_data->g_phi, &hd_data->pitch_phi,
            h_data->n_theta, h_data->n_phi);
    cudaMallocPitch((void**)&hd_data->g_theta, &hd_data->pitch_theta,
            h_data->n_theta, h_data->n_phi);

    // Copy the data across.
    cudaMemcpy2D(hd_data->g_phi, hd_data->pitch_phi, h_data->g_phi,
            h_data->n_theta * sizeof(float2), h_data->n_theta * sizeof(float2),
            h_data->n_phi * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy2D(hd_data->g_theta, hd_data->pitch_theta, h_data->g_theta,
            h_data->n_theta * sizeof(float2), h_data->n_theta * sizeof(float2),
            h_data->n_phi * sizeof(float2), cudaMemcpyHostToDevice);

    // Check for errors.
    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
