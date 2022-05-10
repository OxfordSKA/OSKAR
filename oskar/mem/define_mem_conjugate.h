/* Copyright (c) 2022, The OSKAR Developers. See LICENSE file. */

#define OSKAR_MEM_CONJ(NAME, FP2) KERNEL(NAME) (\
        const unsigned int n, GLOBAL FP2* a)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    FP2 t = a[i];\
    t.y = -t.y;\
    a[i] = t;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
