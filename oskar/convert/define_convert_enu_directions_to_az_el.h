/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_AZ_EL(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, x), GLOBAL_IN(FP, y), GLOBAL_IN(FP, z),\
        GLOBAL_OUT(FP, az_rad), GLOBAL_OUT(FP, el_rad))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const FP x_ = x[i], y_ = y[i], z_ = z[i];\
    const FP r = sqrt(x_*x_ + y_*y_);\
    az_rad[i] = atan2(x_, y_);\
    el_rad[i] = atan2(z_, r);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
