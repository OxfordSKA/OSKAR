/* Copyright (c) 2015-2020, The University of Oxford. See LICENSE file. */

#define OSKAR_AUTO_POWER_MATRIX(NAME, FP, FP2, FP4c) KERNEL(NAME) (const int n,\
        const int offset_in,  GLOBAL_IN(FP4c, jones),\
        const FP src_I, const FP src_Q, const FP src_U, const FP src_V,\
        const int offset_out, GLOBAL_OUT(FP4c, out))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP4c val1, val2, b;\
    OSKAR_LOAD_MATRIX(val1, jones[i + offset_in])\
    val2 = val1;\
    OSKAR_CONSTRUCT_B(FP, b, src_I, src_Q, src_U, src_V)\
    OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, val1, b)\
    OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, val1, val2)\
    out[i + offset_out] = val1;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_AUTO_POWER_SCALAR(NAME, FP, FP2) KERNEL(NAME) (const int n,\
        const int offset_in,  GLOBAL_IN(FP2, jones),\
        const FP src_I, const FP src_Q, const FP src_U, const FP src_V,\
        const int offset_out, GLOBAL_OUT(FP2, out))\
{\
    (void) src_I;\
    (void) src_Q;\
    (void) src_U;\
    (void) src_V;\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP2 val = jones[i + offset_in];\
    val.x = val.x * val.x + val.y * val.y;\
    val.y = 0;\
    out[i + offset_out] = val;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
