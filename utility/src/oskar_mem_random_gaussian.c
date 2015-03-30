/*
 * Copyright (c) 2015, The University of Oxford
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

#include <oskar_mem_random_gaussian.h>
#include <oskar_mem_random_gaussian_cuda.h>
#include <oskar_cuda_check_error.h>
#include <private_random_helpers.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_random_gaussian_f(
        const unsigned int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const float std)
{
    unsigned int i, i4, n1;

    n1 = num_elements / 4;
#pragma omp parallel for private(i, i4)
    for (i = 0; i < n1; ++i)
    {
        float t[4];
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = i * 4;
        oskar_box_muller_f(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_f(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]     = t[0];
        data[i4 + 1] = t[1];
        data[i4 + 2] = t[2];
        data[i4 + 3] = t[3];
    }

    if (num_elements % 4)
    {
        float t[4];
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = n1 * 4;
        oskar_box_muller_f(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_f(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]         = t[0];
        if (i4 + 1 < num_elements)
            data[i4 + 1] = t[1];
        if (i4 + 2 < num_elements)
            data[i4 + 2] = t[2];
        if (i4 + 3 < num_elements)
            data[i4 + 3] = t[3];
    }
}

void oskar_mem_random_gaussian_d(
        const unsigned int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const double std)
{
    unsigned int i, i4, n1;

    n1 = num_elements / 4;
#pragma omp parallel for private(i, i4)
    for (i = 0; i < n1; ++i)
    {
        double t[4];
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = i * 4;
        oskar_box_muller_d(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_d(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]     = t[0];
        data[i4 + 1] = t[1];
        data[i4 + 2] = t[2];
        data[i4 + 3] = t[3];
    }

    if (num_elements % 4)
    {
        double t[4];
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = n1 * 4;
        oskar_box_muller_d(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_d(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]         = t[0];
        if (i4 + 1 < num_elements)
            data[i4 + 1] = t[1];
        if (i4 + 2 < num_elements)
            data[i4 + 2] = t[2];
        if (i4 + 3 < num_elements)
            data[i4 + 3] = t[3];
    }
}

void oskar_mem_random_gaussian(oskar_Mem* data, unsigned int seed,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        double std, int* status)
{
    int type, location;
    size_t num_elements;

    /* Check if safe to proceed. */
    if (*status) return;

    type = oskar_mem_precision(data);
    location = oskar_mem_location(data);
    num_elements = oskar_mem_length(data);
    if (oskar_mem_is_complex(data)) num_elements *= 2;
    if (oskar_mem_is_matrix(data)) num_elements *= 4;

    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_mem_random_gaussian_cuda_f(num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3, (float)std);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_mem_random_gaussian_cuda_d(num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3, std);
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_mem_random_gaussian_f(num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3, (float)std);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_mem_random_gaussian_d(num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3, std);
        }
    }
}

#ifdef __cplusplus
}
#endif
