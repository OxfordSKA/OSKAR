/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_MEM_ADD(NAME, FP) KERNEL(NAME) (\
        const unsigned int off_a, const unsigned int off_b,\
        const unsigned int off_c, const unsigned int n,\
        GLOBAL const FP* a, GLOBAL const FP* b, GLOBAL FP* c)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    c[i + off_c] = a[i + off_a] + b[i + off_b];\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
