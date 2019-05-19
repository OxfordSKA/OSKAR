/* Copyright (c) 2012-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_ELEMENT_TAPER_COSINE_SCALAR(NAME, FP, FP2)\
KERNEL(NAME) (const int n, const FP cos_power, GLOBAL_IN(FP, theta),\
        const int offset_out, GLOBAL_OUT(FP2, jones))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP theta_ = theta[i];\
    const FP cos_theta = (FP) cos(theta_);\
    const FP f = (FP) pow(cos_theta, cos_power);\
    const int i_out = i + offset_out;\
    jones[i_out].x *= f; jones[i_out].y *= f;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_ELEMENT_TAPER_COSINE_MATRIX(NAME, FP, FP4c)\
KERNEL(NAME) (const int n, const FP cos_power, GLOBAL_IN(FP, theta),\
        const int offset_out, GLOBAL_OUT(FP4c, jones))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const FP theta_ = theta[i];\
    const FP cos_theta = (FP) cos(theta_);\
    const FP f = (FP) pow(cos_theta, cos_power);\
    const int i_out = i + offset_out;\
    jones[i_out].a.x *= f; jones[i_out].a.y *= f;\
    jones[i_out].b.x *= f; jones[i_out].b.y *= f;\
    jones[i_out].c.x *= f; jones[i_out].c.y *= f;\
    jones[i_out].d.x *= f; jones[i_out].d.y *= f;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
