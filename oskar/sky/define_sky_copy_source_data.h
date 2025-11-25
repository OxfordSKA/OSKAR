/* Copyright (c) 2017-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_SKY_COPY_SOURCE_DATA(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(int, mask), GLOBAL_IN(int, indices),\
        const int num_col, GLOBAL const FP** col_in, GLOBAL FP** col_out\
)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    if (mask[i]) {\
        const int i_out = indices[i];\
        _Pragma("unroll")\
        for (int c = 0; c < num_col; ++c) {\
            (col_out[c])[i_out] = (col_in[c])[i];\
        }\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
