/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_STATION_UVW_TO_BASELINE_UVW(NAME, FP) KERNEL(NAME) (\
        const int num, const int offset_in, GLOBAL_IN(FP, u),\
        GLOBAL_IN(FP, v), GLOBAL_IN(FP, w), const int offset_out,\
        GLOBAL_OUT(FP, uu), GLOBAL_OUT(FP, vv), GLOBAL_OUT(FP, ww))\
{\
    const int s1 = GROUP_ID_X;\
    const int i1 = s1 + offset_in;\
    for (int s2 = s1 + LOCAL_ID_X + 1; s2 < num; s2 += LOCAL_DIM_X) {\
        const int i2 = s2 + offset_in;\
        const int b = s1 * (num - 1) - (s1 - 1) * s1/2 + s2 - s1 - 1 + offset_out;\
        uu[b] = u[i2] - u[i1]; vv[b] = v[i2] - v[i1]; ww[b] = w[i2] - w[i1];\
    }\
}\
OSKAR_REGISTER_KERNEL(NAME)
