/* Copyright (c) 2017-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_SKY_COPY_SOURCE_DATA(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(int, mask), GLOBAL_IN(int, indices),\
        const int num_col, GLOBAL_IN(FP, col_in), GLOBAL_OUT(FP, col_out)\
)\
{\
    /* Casts to try and get OpenCL version working, which does not support */\
    /* pointers-to-pointers in kernel arguments. */\
    GLOBAL const FP* const* col_in_ = (const FP* const*) col_in;\
    GLOBAL FP** col_out_ = (FP**) col_out;\
    KERNEL_LOOP_X(int, i, 0, num)\
    if (mask[i]) {\
        const int i_out = indices[i];\
        _Pragma("unroll")\
        for (int c = 0; c < num_col; ++c) {\
            (col_out_[c])[i_out] = (col_in_[c])[i];\
        }\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
