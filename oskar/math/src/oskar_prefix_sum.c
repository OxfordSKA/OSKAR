/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_prefix_sum.h"
#include "utility/oskar_device.h"

static size_t get_block_size(size_t num_elements)
{
    if (num_elements == 0) { return 0; }
    else if (num_elements <= 1) { return 1; }
    else if (num_elements <= 2) { return 2; }
    else if (num_elements <= 4) { return 4; }
    else if (num_elements <= 8) { return 8; }
    else if (num_elements <= 16) { return 16; }
    else if (num_elements <= 32) { return 32; }
    else if (num_elements <= 64) { return 64; }
    else if (num_elements <= 128) { return 128; }
    else if (num_elements <= 256) { return 256; }
    return 512;
}


void oskar_prefix_sum(size_t num_elements, const oskar_Mem* in,
        oskar_Mem* out, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(in);
    const int location = oskar_mem_location(in);
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
    if (oskar_mem_length(out) < num_elements + 1)
    {
        /* Last element is total number. */
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        size_t i = 0;
        int sum = 0, *out_ = 0;
        const int* in_ = oskar_mem_int_const(in, status);
        out_ = oskar_mem_int(out, status);
        for (i = 0; i < num_elements; ++i)
        {
            int x = in_[i];
            out_[i] = sum;
            sum += x;
        }
        out_[i] = sum;
    }
    else
    {
        size_t local_size[] = {32, 1, 1}, global_size[] = {32, 1, 1};
        size_t arg_size_local[] = {0};
        int num_local_args = 0;
        const int num = (int) num_elements;
        const oskar_Arg args[] = {
                {INT_SZ, &num},
                {PTR_SZ, oskar_mem_buffer_const(in)},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        if (oskar_device_is_gpu(location))
        {
            local_size[0] = get_block_size(num + 1);
            oskar_device_check_local_size(location, 0, local_size);
            arg_size_local[0] = 2 * local_size[0] * sizeof(int);
            global_size[0] = local_size[0]; /* Only 1 block. */
            num_local_args = 1;
        }
        oskar_device_launch_kernel("prefix_sum_int", location, 1,
                local_size, global_size, sizeof(args) / sizeof(oskar_Arg),
                args, num_local_args, arg_size_local, status);
    }
}
