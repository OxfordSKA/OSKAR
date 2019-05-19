/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_MEM_SCALE_REAL(NAME, FP) KERNEL(NAME) (const unsigned int offset,\
        const unsigned int n, const FP val, GLOBAL FP* a)\
{\
    KERNEL_LOOP_X(unsigned int, i, 0, n)\
    a[i + offset] *= val;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
