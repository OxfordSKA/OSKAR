/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_MEM_MUL_RR_R(NAME, FP) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP* a, GLOBAL const FP* b, GLOBAL FP* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    c[i + off_c] = a[i + off_a] * b[i + off_b];\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_MUL_CC_C(NAME, FP2) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP2* a, GLOBAL const FP2* b, GLOBAL FP2* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP2 cc;\
    const FP2 ac = a[i + off_a];\
    const FP2 bc = b[i + off_b];\
    OSKAR_MUL_COMPLEX(cc, ac, bc)\
    c[i + off_c] = cc;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_MUL_CC_M(NAME, FP, FP2, FP4c) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP2* a, GLOBAL const FP2* b, GLOBAL FP4c* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP2 cc;\
    const FP2 ac = a[i + off_a];\
    const FP2 bc = b[i + off_b];\
    OSKAR_MUL_COMPLEX(cc, ac, bc)\
    FP4c m;\
    m.a = cc;\
    MAKE_ZERO2(FP, m.b);\
    MAKE_ZERO2(FP, m.c);\
    m.d = cc;\
    c[i + off_c] = m;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_MUL_CM_M(NAME, FP2, FP4c) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP2* a, GLOBAL const FP4c* b, GLOBAL FP4c* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    const FP2 ac = a[i + off_a];\
    FP4c bc = b[i + off_b];\
    OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR_IN_PLACE(FP2, bc, ac)\
    c[i + off_c] = bc;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_MUL_MC_M(NAME, FP2, FP4c) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP4c* a, GLOBAL const FP2* b, GLOBAL FP4c* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP4c ac = a[i + off_a];\
    const FP2 bc = b[i + off_b];\
    OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR_IN_PLACE(FP2, ac, bc)\
    c[i + off_c] = ac;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_MUL_MM_M(NAME, FP2, FP4c) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP4c* a, GLOBAL const FP4c* b, GLOBAL FP4c* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP4c ac = a[i + off_a];\
    const FP4c bc = b[i + off_b];\
    OSKAR_MUL_COMPLEX_MATRIX_IN_PLACE(FP2, ac, bc)\
    c[i + off_c] = ac;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
