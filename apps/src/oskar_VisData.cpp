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

#include "apps/oskar_VisData.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

oskar_VisData::oskar_VisData(const unsigned num_stations, const unsigned num_vis_dumps)
{
    _num_baselines = num_stations * (num_stations - 1) / 2;
    _num_vis_coordinates = _num_baselines * num_vis_dumps;

//    printf("= allocating visibility data arrays for:\n");
//    printf("   - num vis dumps       = %i\n", num_vis_dumps);
//    printf("   - num stations        = %i\n", num_stations);
//    printf("   - num vis coordinates = %i\n", _num_vis_coordinates);

    _u.resize(_num_vis_coordinates);
    _v.resize(_num_vis_coordinates);
    _w.resize(_num_vis_coordinates);
    _vis.resize(_num_vis_coordinates);
}

oskar_VisData::~oskar_VisData()
{}

void oskar_VisData::write(const char* filename)
{
    FILE * file;
    file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        return;
    }
    for (unsigned i = 0; i < _num_vis_coordinates; ++i)
    {
        fwrite(&_u[i],     sizeof(double), 1, file);
        fwrite(&_v[i],     sizeof(double), 1, file);
        fwrite(&_w[i],     sizeof(double), 1, file);
        fwrite(&_vis[i].x, sizeof(double), 1, file);
        fwrite(&_vis[i].y, sizeof(double), 1, file);
//        printf("%f %f %f %f %f\n", _u[i], _v[i], _w[i], _vis[i].x, _vis[i].y);

    }
    fclose(file);
}


void oskar_VisData::load(const char* filename)
{
    FILE * file;
    file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "ERROR: Failed to open input visibility data file.\n");
        return;
    }

    _u.clear();
    _v.clear();
    _w.clear();
    _vis.clear();

    size_t record_size = 5 * sizeof(double);
    double* buffer = (double*) malloc(record_size);
    while (fread(buffer, record_size, 1, file) == 1)
    {
        _u.push_back(buffer[0]);
        _v.push_back(buffer[1]);
        _w.push_back(buffer[2]);
        _vis.push_back(*((double2*)(buffer + 3)));
//        printf("%f %f %f %f %f\n", _u.back(), _v.back(), _w.back(), _vis.back().x, _vis.back().y);
    }

    fclose(file);

    _num_vis_coordinates = _u.size();
}

