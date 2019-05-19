/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_GAUSSIAN_CIRCULAR_COMPLEX(NAME, FP, FP2) KERNEL(NAME) (\
        const int n, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y),\
        const FP inv_2_var, GLOBAL_OUT(FP2, z))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP x_ = x[i]; const FP y_ = y[i];\
    const FP arg = -(x_ * x_ + y_ * y_) * inv_2_var;\
    z[i].x = exp(arg); z[i].y = (FP) 0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_GAUSSIAN_CIRCULAR_MATRIX(NAME, FP, FP4c) KERNEL(NAME) (\
        const int n, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y),\
        const FP inv_2_var, GLOBAL_OUT(FP4c, z))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP x_ = x[i]; const FP y_ = y[i];\
    const FP arg = -(x_ * x_ + y_ * y_) * inv_2_var;\
    const FP value = exp(arg);\
    z[i].a.x = value; z[i].a.y = (FP) 0;\
    MAKE_ZERO2(FP, z[i].b);\
    MAKE_ZERO2(FP, z[i].c);\
    z[i].d.x = value; z[i].d.y = (FP) 0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
