/* Copyright (c) 2017-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_SKY_COPY_SOURCE_DATA(NAME, FP) KERNEL(NAME) (\
        const int num, const int capacity_in, const int capacity_out,\
        GLOBAL_IN(int, mask), GLOBAL_IN(int, indices),\
        const int num_col, GLOBAL_IN(FP, table_in), GLOBAL_OUT(FP, table_out)\
)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    if (mask[i]) {\
        const int i_out = indices[i];\
        _Pragma("unroll")\
        for (int c = 0; c < num_col; ++c) {\
            table_out[c * capacity_out + i_out] =\
                    table_in[c * capacity_in + i];\
        }\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
