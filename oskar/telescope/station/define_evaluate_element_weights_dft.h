/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_ELEMENT_WEIGHTS_DFT(NAME, FP, FP2) KERNEL(NAME) (\
        const int n, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y), GLOBAL_IN(FP, z),\
        GLOBAL_IN(FP, cable_length_error), const FP wavenumber,\
        const FP x1, const FP y1, const FP z1, GLOBAL_OUT(FP2, weights))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP re, im;\
    const FP p = wavenumber * (\
            x[i] * x1 + y[i] * y1 + z[i] * z1 + cable_length_error[i]);\
    SINCOS(-p, im, re);\
    weights[i].x = re; weights[i].y = im;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
