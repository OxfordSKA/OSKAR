/* Copyright (c) 2020, The OSKAR Developers. See LICENSE file. */

#define OSKAR_CONVERT_REL_DIR_TO_LON_LAT(NAME, IS_3D, FP) KERNEL_PUB(NAME) (\
        const int num,\
        GLOBAL const FP* l,\
        GLOBAL const FP* m,\
        GLOBAL const FP* n,\
        const FP lon0_rad,\
        const FP cos_lat0,\
        const FP sin_lat0,\
        GLOBAL FP* lon_rad,\
        GLOBAL FP* lat_rad)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const FP l_ = l[i], m_ = m[i];\
    const FP n_ = (IS_3D) ? n[i] : sqrt((FP)1 - l_*l_ - m_*m_);\
    lat_rad[i] = asin(n_ * sin_lat0 + m_ * cos_lat0);\
    lon_rad[i] = lon0_rad + atan2(l_, cos_lat0 * n_ - m_ * sin_lat0);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
