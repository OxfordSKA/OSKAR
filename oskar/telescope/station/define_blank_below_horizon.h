/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_BLANK_BELOW_HORIZON_SCALAR(NAME, FP, FP2)\
KERNEL(NAME) (const int offset_mask, const int n, GLOBAL_IN(FP, mask),\
        const int offset_out, GLOBAL_OUT(FP2, jones))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const int i_out = offset_out + i;\
    if (mask[i + offset_mask] < (FP)0)\
        MAKE_ZERO2(FP, jones[i_out]);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_BLANK_BELOW_HORIZON_MATRIX(NAME, FP, FP4c)\
KERNEL(NAME) (const int offset_mask, const int n, GLOBAL_IN(FP, mask),\
        const int offset_out, GLOBAL_OUT(FP4c, jones))\
{\
    FP4c zero;\
    MAKE_ZERO2(FP, zero.a);\
    MAKE_ZERO2(FP, zero.b);\
    MAKE_ZERO2(FP, zero.c);\
    MAKE_ZERO2(FP, zero.d);\
    KERNEL_LOOP_X(int, i, 0, n)\
    const int i_out = offset_out + i;\
    if (mask[i + offset_mask] < (FP)0) jones[i_out] = zero;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
