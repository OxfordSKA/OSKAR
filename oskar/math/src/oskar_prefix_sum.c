/*
 * Copyright (c) 2017, The University of Oxford
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

#include "math/oskar_prefix_sum.h"
#include "math/oskar_prefix_sum_cuda.h"

static size_t get_block_size(size_t num_elements)
{
    if (num_elements == 0) return 0;
    else if (num_elements <= 1) return 1;
    else if (num_elements <= 2) return 2;
    else if (num_elements <= 4) return 4;
    else if (num_elements <= 8) return 8;
    else if (num_elements <= 16) return 16;
    else if (num_elements <= 32) return 32;
    else if (num_elements <= 64) return 64;
    else if (num_elements <= 128) return 128;
    return 256;
}


void oskar_prefix_sum(size_t num_elements, const oskar_Mem* in,
        oskar_Mem* out, int init_val, int exclusive, int* status)
{
    int type, location;
    if (*status) return;

    /* Check data type and location. */
    type = oskar_mem_type(in);
    location = oskar_mem_location(in);
    if (location != oskar_mem_location(out))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (type != oskar_mem_type(out))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (type != OSKAR_INT)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Prefix sum. */
    if (location == OSKAR_CPU)
    {
        size_t i;
        int sum = 0, *out_;
        const int* in_ = oskar_mem_int_const(in, status);
        out_ = oskar_mem_int(out, status);
        if (exclusive)
        {
            sum = init_val;
            for (i = 0; i < num_elements; ++i)
            {
                int x = in_[i];
                out_[i] = sum;
                sum += x;
            }
        }
        else
        {
            sum = in_[0];
            out_[0] = sum;
            for (i = 1; i < num_elements; ++i)
            {
                sum += in_[i];
                out_[i] = sum;
            }
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        oskar_Mem* block_sums;
        size_t block_size, num_blocks;
        block_size = get_block_size(num_elements);
        num_blocks = (num_elements + block_size - 1) / block_size;

        /* Allocate memory to hold a sum per block. */
        block_sums = oskar_mem_create(type, location, num_blocks, status);

        /* Local scan. */
        oskar_prefix_sum_cuda_int((int) num_elements,
                oskar_mem_int_const(in, status), oskar_mem_int(out, status),
                (int) num_blocks, (int) block_size,
                oskar_mem_int(block_sums, status), init_val, exclusive);

        if (num_blocks > 1)
        {
            /* Inclusive scan block sums. */
            oskar_prefix_sum(num_blocks, block_sums, block_sums,
                    init_val, 0, status);

            /* Add block sums to each block. */
            oskar_prefix_sum_finalise_cuda_int((int) num_elements,
                    oskar_mem_int(out, status), (int) num_blocks,
                    (int) block_size, oskar_mem_int_const(block_sums, status),
                    (int) block_size);
        }

        /* Clean up. */
        oskar_mem_free(block_sums, status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}
