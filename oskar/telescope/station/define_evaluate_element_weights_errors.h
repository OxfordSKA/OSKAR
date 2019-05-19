/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_ELEMENT_WEIGHTS_ERR(NAME, FP, FP2)\
KERNEL(NAME) (const int n, GLOBAL_IN(FP, amp_gain),\
        GLOBAL_IN(FP, amp_error), GLOBAL_IN(FP, phase_offset),\
        GLOBAL_IN(FP, phase_error), GLOBAL_OUT(FP2, errors))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP re, im; FP2 r = errors[i];\
    r.x *= amp_error[i];   r.x += amp_gain[i];\
    r.y *= phase_error[i]; r.y += phase_offset[i];\
    SINCOS(r.y, im, re);\
    errors[i].x = re * r.x; errors[i].y = im * r.x;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
