/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_ECEF_TO_STATION_UVW(NAME, FP) KERNEL(NAME) (\
        const int num,\
        GLOBAL_IN(FP, x),\
        GLOBAL_IN(FP, y),\
        GLOBAL_IN(FP, z),\
        const FP sin_ha0,\
        const FP cos_ha0,\
        const FP sin_dec0,\
        const FP cos_dec0,\
        const int ignore_w_components,\
        const int offset_out,\
        GLOBAL_OUT(FP, u),\
        GLOBAL_OUT(FP, v),\
        GLOBAL_OUT(FP, w))\
{\
    FP zero;\
    MAKE_ZERO(FP, zero);\
    KERNEL_LOOP_X(int, i, 0, num)\
    const FP x_ = x[i], y_ = y[i], z_ = z[i];\
    const FP t = x_ * cos_ha0 - y_ * sin_ha0;\
    const int j = i + offset_out;\
    u[j] = x_ * sin_ha0 + y_ * cos_ha0;\
    v[j] = z_ * cos_dec0 - t * sin_dec0;\
    w[j] = (ignore_w_components ? zero : z_ * sin_dec0 + t * cos_dec0);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
