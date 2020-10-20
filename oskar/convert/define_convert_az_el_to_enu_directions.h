/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_AZ_EL_TO_ENU_DIR(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, az_rad), GLOBAL_IN(FP, el_rad),\
        GLOBAL_OUT(FP, x), GLOBAL_OUT(FP, y), GLOBAL_OUT(FP, z))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_az, cos_az, sin_el, cos_el;\
    SINCOS(az_rad[i], sin_az, cos_az);\
    SINCOS(el_rad[i], sin_el, cos_el);\
    x[i] = cos_el * sin_az; /* East */\
    y[i] = cos_el * cos_az; /* North */\
    z[i] = sin_el; /* Up */\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
