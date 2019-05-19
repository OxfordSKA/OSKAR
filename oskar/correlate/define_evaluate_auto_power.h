/* Copyright (c) 2015-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_AUTO_POWER_MATRIX(NAME, FP2, FP4c) KERNEL(NAME) (const int n,\
        const int offset_in,  GLOBAL_IN(FP4c, jones),\
        const int offset_out, GLOBAL_OUT(FP4c, out))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP4c val1, val2;\
    OSKAR_LOAD_MATRIX(val1, jones[i + offset_in])\
    val2 = val1;\
    OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, val1, val2)\
    out[i + offset_out] = val1;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_AUTO_POWER_SCALAR(NAME, FP2) KERNEL(NAME) (const int n,\
        const int offset_in,  GLOBAL_IN(FP2, jones),\
        const int offset_out, GLOBAL_OUT(FP2, out))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP2 val = jones[i + offset_in];\
    val.x = val.x * val.x + val.y * val.y;\
    val.y = 0;\
    out[i + offset_out] = val;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
