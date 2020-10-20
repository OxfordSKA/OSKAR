/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_AZ_EL_TO_RA_DEC(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, az_rad), GLOBAL_IN(FP, el_rad),\
        const FP lst_rad, const FP cos_lat, const FP sin_lat,\
        GLOBAL_OUT(FP, ra_rad), GLOBAL_OUT(FP, dec_rad))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_az, cos_az, sin_el, cos_el;\
    SINCOS(az_rad[i], sin_az, cos_az);\
    SINCOS(el_rad[i], sin_el, cos_el);\
    const FP y = -cos_el * sin_az;\
    const FP x = sin_el * cos_lat - cos_el * cos_az * sin_lat;\
    const FP z = sin_el * sin_lat + cos_el * cos_az * cos_lat;\
    const FP ha_rad = atan2(y, x);\
    ra_rad[i] = lst_rad - ha_rad;\
    dec_rad[i] = asin(z);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
