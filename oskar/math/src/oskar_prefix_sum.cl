kernel void prefix_sum_int(
        const int num_elements,
        global const int* restrict in,
        global int* restrict out,
        global int* restrict block_sums,
        local int* restrict scratch,
        const int init_val,
        const int exclusive)
{
    const int i = get_global_id(0);
    const int block_size = get_local_size(0);
    int pos = get_local_id(0);
    // Copy input data to local memory.
    scratch[pos] = 0;
    pos += block_size;
    scratch[pos] = 0;
    if (i < num_elements) {
        if (exclusive) {
            const int local_init = (i == 0) ? init_val : 0;
            scratch[pos] = (get_local_id(0) > 0) ? in[i - 1] : local_init;
        }
        else {
            scratch[pos] = in[i];
        }
    }
    // Prefix sum.
    for (int j = 1; j < block_size; j <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const int x = scratch[pos - j];
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[pos] += x;
    }
    // Store local results.
    if (i < num_elements) out[i] = scratch[pos];
    // Store sum for the block.
    if (get_local_id(0) == get_local_size(0) - 1) {
        const int x = (i < num_elements) ? in[i] : 0;
        block_sums[get_group_id(0)] = exclusive ?
                x + scratch[pos] : scratch[pos];
    }
}

kernel void prefix_sum_finalise_int(
        const int num_elements,
        global int* restrict out,
        global const int* restrict block_sums,
        int offset)
{
    const int i = get_global_id(0) + offset;
    if (i < num_elements) out[i] += block_sums[get_group_id(0)];
}

kernel void prefix_sum_cpu_int(
        const int num_elements,
        global const int* restrict in,
        global int* restrict out,
        const int init_val,
        const int exclusive)
{
    if (get_global_id(0) == 0)
    {
        if (exclusive)
        {
            int sum = init_val;
            for (int i = 0; i < num_elements; ++i)
            {
                int x = in[i];
                out[i] = sum;
                sum += x;
            }
        }
        else
        {
            int sum = in[0];
            out[0] = sum;
            for (int i = 1; i < num_elements; ++i)
            {
                sum += in[i];
                out[i] = sum;
            }
        }
    }
}
