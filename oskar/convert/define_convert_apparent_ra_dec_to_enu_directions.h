/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_RA_DEC_TO_ENU_DIR(NAME, FP) KERNEL_PUB(NAME) (\
        const int num,\
        GLOBAL_IN(FP, ra_rad),\
        GLOBAL_IN(FP, dec_rad),\
        const FP lst_rad,\
        const FP sin_lat,\
        const FP cos_lat,\
        const int offset_out,\
        GLOBAL_OUT(FP, x),\
        GLOBAL_OUT(FP, y),\
        GLOBAL_OUT(FP, z))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_ha, cos_ha, sin_dec, cos_dec;\
    const FP ha_rad = lst_rad - ra_rad[i];\
    SINCOS(ha_rad, sin_ha, cos_ha);\
    SINCOS(dec_rad[i], sin_dec, cos_dec);\
    const int i_out = i + offset_out;\
    x[i_out] = -cos_dec * sin_ha;\
    y[i_out] = cos_lat * sin_dec - sin_lat * cos_dec * cos_ha;\
    z[i_out] = sin_lat * sin_dec + cos_lat * cos_dec * cos_ha;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
