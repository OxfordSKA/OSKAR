/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_REL_DIR(NAME, FP) KERNEL(NAME) (\
        const int offset_in, const int num, GLOBAL_IN(FP, x),\
        GLOBAL_IN(FP, y), GLOBAL_IN(FP, z),\
        const FP sin_ha0, const FP cos_ha0, const FP sin_dec0,\
        const FP cos_dec0, const FP sin_lat, const FP cos_lat,\
        const int offset_out, GLOBAL_OUT(FP, l), GLOBAL_OUT(FP, m),\
        GLOBAL_OUT(FP, n))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const int i_in = i + offset_in, i_out = i + offset_out;\
    FP x_ = x[i_in], y_ = y[i_in], z_ = z[i_in], t;\
    l[i_out] = x_ * cos_ha0 -\
            y_ * sin_ha0 * sin_lat +\
            z_ * sin_ha0 * cos_lat;\
    t = sin_dec0 * cos_ha0;\
    m[i_out] = x_ * sin_dec0 * sin_ha0 +\
            y_ * (cos_dec0 * cos_lat + t * sin_lat) +\
            z_ * (cos_dec0 * sin_lat - t * cos_lat);\
    t = cos_dec0 * cos_ha0;\
    n[i_out] = -x_ * cos_dec0 * sin_ha0 +\
            y_ * (sin_dec0 * cos_lat - t * sin_lat) +\
            z_ * (sin_dec0 * sin_lat + t * cos_lat);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
