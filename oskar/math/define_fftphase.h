/* Copyright (c) 2016-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_FFTPHASE(NAME, FP) KERNEL(NAME) (\
        const int num_x, const int num_y, GLOBAL FP* complex_data)\
{\
    KERNEL_LOOP_Y(int, iy, 0, num_y)\
    KERNEL_LOOP_X(int, ix, 0, num_x)\
    const int x = 1 - (((ix + iy) & 1) << 1);\
    complex_data[((iy * num_x + ix) << 1)]     *= x;\
    complex_data[((iy * num_x + ix) << 1) + 1] *= x;\
    KERNEL_LOOP_END\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
