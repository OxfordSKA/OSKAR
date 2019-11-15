/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_LON_LAT_TO_REL_DIR(NAME, IS_3D, FP) KERNEL_PUB(NAME) (\
        const int num, GLOBAL_IN(FP, lon_rad), GLOBAL_IN(FP, lat_rad),\
        const FP lon0_rad, const FP cos_lat0, const FP sin_lat0,\
        GLOBAL_OUT(FP, l), GLOBAL_OUT(FP, m), GLOBAL_OUT(FP, n))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_lon, cos_lon, sin_lat, cos_lat;\
    const FP lon = lon_rad[i] - lon0_rad, lat = lat_rad[i];\
    SINCOS(lon, sin_lon, cos_lon);\
    SINCOS(lat, sin_lat, cos_lat);\
    l[i] = cos_lat * sin_lon;\
    m[i] = cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_lon;\
    if (IS_3D) n[i] = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_lon;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
