/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_LOCAL(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y), GLOBAL_IN(FP, z),\
        const FP cos_phi, const FP sin_phi, const FP cos_theta,\
        const FP sin_theta, GLOBAL_OUT(FP, l), GLOBAL_OUT(FP, m))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const FP x_ = x[i], y_ = y[i], z_ = z[i];\
    l[i] = x_ * cos_phi - y_ * sin_phi;\
    m[i] = x_ * cos_theta * sin_phi + y_ * cos_theta * cos_phi - z_ * sin_theta;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
