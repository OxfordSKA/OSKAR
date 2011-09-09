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

#include "interferometry/oskar_VisData.h"

#include "stdlib.h"
#include "string.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_allocate_vis_data_d(const unsigned num_samples, oskar_VisData_d* vis)
{
    vis->num_samples = num_samples;

    vis->u      = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->v      = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->w      = (double*)  malloc(vis->num_samples * sizeof(double));
    vis->amp    = (double2*) malloc(vis->num_samples * sizeof(double2));

    memset(vis->u,      0, vis->num_samples * sizeof(double));
    memset(vis->v,      0, vis->num_samples * sizeof(double));
    memset(vis->w,      0, vis->num_samples * sizeof(double));
    memset(vis->amp,    0, vis->num_samples * sizeof(double2));
}

void oskar_allocate_vis_data_f(const unsigned num_samples, oskar_VisData_f* vis)
{
    vis->num_samples = num_samples;

    vis->u      = (float*)  malloc(vis->num_samples * sizeof(float));
    vis->v      = (float*)  malloc(vis->num_samples * sizeof(float));
    vis->w      = (float*)  malloc(vis->num_samples * sizeof(float));
    vis->amp    = (float2*) malloc(vis->num_samples * sizeof(float2));

    memset(vis->u,      0, vis->num_samples * sizeof(float));
    memset(vis->v,      0, vis->num_samples * sizeof(float));
    memset(vis->w,      0, vis->num_samples * sizeof(float));
    memset(vis->amp,    0, vis->num_samples * sizeof(float2));
}



void oskar_free_vis_data_d(oskar_VisData_d* vis)
{
    vis->num_samples = 0;
    free(vis->u);
    free(vis->v);
    free(vis->w);
    free(vis->amp);
}

void oskar_free_vis_data_f(oskar_VisData_f* vis)
{
    vis->num_samples = 0;
    free(vis->u);
    free(vis->v);
    free(vis->w);
    free(vis->amp);
}


void oskar_write_vis_data_d(const char* filename, const oskar_VisData_d* vis)
{
    FILE* file;
    file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        return;
    }
    for (int i = 0; i < vis->num_samples; ++i)
    {
        fwrite(&(vis->u[i]),     sizeof(double), 1, file);
        fwrite(&(vis->v[i]),     sizeof(double), 1, file);
        fwrite(&(vis->w[i]),     sizeof(double), 1, file);
        fwrite(&(vis->amp[i].x), sizeof(double), 1, file);
        fwrite(&(vis->amp[i].y), sizeof(double), 1, file);

    }
    fclose(file);
}

void oskar_load_vis_data_d(const char* filename, oskar_VisData_d* vis)
{
    FILE* file;
    file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open input visibility data file.\n");
        return;
    }
    vis->num_samples = 0;
    vis->u = NULL;
    vis->v = NULL;
    vis->w = NULL;
    vis->amp = NULL;

    size_t record_size = 5 * sizeof(double);
    double* buffer = (double*) malloc(record_size);
    while (fread(buffer, record_size, 1, file) == 1)
    {
        // Ensure enough space in arrays.
        if (vis->num_samples % 100 == 0)
        {
            size_t mem_size = (vis->num_samples + 100) * sizeof(double);
            vis->u = (double*) realloc(vis->u, mem_size);
            vis->v = (double*) realloc(vis->v, mem_size);
            vis->w = (double*) realloc(vis->w, mem_size);
            mem_size = (vis->num_samples + 100) * sizeof(double2);
            vis->amp = (double2*) realloc(vis->amp, mem_size);
        }
        vis->u[vis->num_samples]     = buffer[0];
        vis->v[vis->num_samples]     = buffer[1];
        vis->w[vis->num_samples]     = buffer[2];
        vis->amp[vis->num_samples].x = buffer[3];
        vis->amp[vis->num_samples].y = buffer[4];
        vis->num_samples++;
    }

    fclose(file);
}



void oskar_write_vis_data_f(const char* filename, const oskar_VisData_f* vis)
{
    FILE* file;
    file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        return;
    }
    for (int i = 0; i < vis->num_samples; ++i)
    {
        fwrite(&(vis->u[i]),     sizeof(float), 1, file);
        fwrite(&(vis->v[i]),     sizeof(float), 1, file);
        fwrite(&(vis->w[i]),     sizeof(float), 1, file);
        fwrite(&(vis->amp[i].x), sizeof(float), 1, file);
        fwrite(&(vis->amp[i].y), sizeof(float), 1, file);

    }
    fclose(file);
}

void oskar_load_vis_data_f(const char* filename, oskar_VisData_f* vis)
{
    FILE* file;
    file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open input visibility data file.\n");
        return;
    }
    vis->num_samples = 0;
    vis->u   = NULL;
    vis->v   = NULL;
    vis->w   = NULL;
    vis->amp = NULL;

    size_t record_size = 5 * sizeof(float);
    float* buffer = (float*) malloc(record_size);
    while (fread(buffer, record_size, 1, file) == 1)
    {
        // Ensure enough space in arrays.
        if (vis->num_samples % 100 == 0)
        {
            size_t mem_size = (vis->num_samples + 100) * sizeof(float);
            vis->u = (float*) realloc(vis->u, mem_size);
            vis->v = (float*) realloc(vis->v, mem_size);
            vis->w = (float*) realloc(vis->w, mem_size);
            mem_size = (vis->num_samples + 100) * sizeof(float2);
            vis->amp = (float2*) realloc(vis->amp, mem_size);
        }
        vis->u[vis->num_samples]     = buffer[0];
        vis->v[vis->num_samples]     = buffer[1];
        vis->w[vis->num_samples]     = buffer[2];
        vis->amp[vis->num_samples].x = buffer[3];
        vis->amp[vis->num_samples].y = buffer[4];
        vis->num_samples++;
    }
    fclose(file);
}




#ifdef __cplusplus
}
#endif

