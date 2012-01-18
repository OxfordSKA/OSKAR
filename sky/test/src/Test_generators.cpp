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

#include "sky/test/Test_generators.h"
#include <cuda_runtime_api.h>

#include "sky/oskar_generate_random_coordinate.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

void Test_generators::test_random_coordinates()
{
    int num_sources = 10000;
    double* ra = (double*)malloc(num_sources * sizeof(double));
    double* dec = (double*)malloc(num_sources * sizeof(double));

    srand(time(NULL));

    for (int i = 0; i < num_sources; ++i)
    {
        oskar_generate_random_coordinate(&ra[i], &dec[i]);
    }

//    FILE* file = fopen("temp_coords.dat", "wb");
//    for (int i = 0; i < num_sources; ++i)
//    {
//        fwrite((const void*)&ra[i], sizeof(double), 1, file);
//        fwrite((const void*)&dec[i], sizeof(double), 1, file);
//    }
//    fclose(file);

    // Matlab code to plot the results:
    //      fid = fopen('temp_coords.dat');
    //      coords = fread(fid, [2 10000], 'double');
    //      [x y z] = sph2cart(coords(1, :), coords(2,:), 1);
    //      scatter3(x,y,z,10);
    //      fclose(fid);

    free(ra);
    free(dec);

}
