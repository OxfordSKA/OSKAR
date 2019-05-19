/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_MEM_SET_VALUE_REAL_REAL(NAME, FP) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n, const FP val,\
        GLOBAL FP* a)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    a[i + offset] = val;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_SET_VALUE_REAL_COMPLEX(NAME, FP, FP2) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n, const FP val,\
        GLOBAL FP2* a)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP2 v; v.x = val; v.y = (FP) 0;\
    a[i + offset] = v;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_MEM_SET_VALUE_REAL_MATRIX(NAME, FP, FP4c) KERNEL(NAME) (\
        const unsigned int offset, const unsigned int n, const FP val,\
        GLOBAL FP4c* a)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP4c v;\
    v.a.x = val; v.a.y = (FP) 0;\
    MAKE_ZERO2(FP, v.b);\
    MAKE_ZERO2(FP, v.c);\
    v.d.x = val; v.d.y = (FP) 0;\
    a[i + offset] = v;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
