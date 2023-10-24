/* Copyright (c) 2023, The OSKAR Developers. See LICENSE file. */

#define OSKAR_GAUSSIAN_ELLIPSE_COMPLEX(NAME, FP, FP2) KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, l),\
        GLOBAL_IN(FP, m),\
        const FP a,\
        const FP b,\
        const FP c,\
        GLOBAL_OUT(FP2, z))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP l_ = l[i];\
    const FP m_ = m[i];\
    const FP arg = -(a * l_ * l_ + 2 * b * l_ * m_ + c * m_ * m_);\
    const FP value = exp(arg);\
    z[i].x = value; z[i].y = (FP) 0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_GAUSSIAN_ELLIPSE_MATRIX(NAME, FP, FP4c) KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, l),\
        GLOBAL_IN(FP, m),\
        const FP x_a,\
        const FP x_b,\
        const FP x_c,\
        const FP y_a,\
        const FP y_b,\
        const FP y_c,\
        GLOBAL_OUT(FP4c, z))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP l_ = l[i];\
    const FP m_ = m[i];\
    const FP arg_x = -(x_a * l_ * l_ + 2 * x_b * l_ * m_ + x_c * m_ * m_);\
    const FP value_x = exp(arg_x);\
    const FP arg_y = -(y_a * l_ * l_ + 2 * y_b * l_ * m_ + y_c * m_ * m_);\
    const FP value_y = exp(arg_y);\
    z[i].a.x = value_x; z[i].a.y = (FP) 0;\
    z[i].d.x = value_y; z[i].d.y = (FP) 0;\
    MAKE_ZERO2(FP, z[i].b);\
    MAKE_ZERO2(FP, z[i].c);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
