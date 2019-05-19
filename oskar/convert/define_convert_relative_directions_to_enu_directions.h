/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_REL_DIR_TO_ENU_DIR(NAME, FP) KERNEL(NAME) (\
        const int at_origin, const int bypass, const int offset_in,\
        const int num, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m), GLOBAL_IN(FP, n),\
        const FP sin_ha0, const FP cos_ha0, const FP sin_dec0,\
        const FP cos_dec0, const FP sin_lat, const FP cos_lat,\
        const int offset_out, GLOBAL_OUT(FP, x), GLOBAL_OUT(FP, y),\
        GLOBAL_OUT(FP, z))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const int i_in = i + offset_in, i_out = i + offset_out;\
    FP l_, m_, n_, t;\
    if (at_origin) {\
        l_ = (FP)0; m_ = (FP)0; n_ = (FP)1;\
    } else {\
        l_ = l[i_in]; m_ = m[i_in]; n_ = n[i_in];\
    }\
    if (bypass) {\
        x[i_out] = l_; y[i_out] = m_; z[i_out] = n_;\
    } else {\
        x[i_out] = l_ * cos_ha0 +\
                m_ * sin_ha0 * sin_dec0 -\
                n_ * sin_ha0 * cos_dec0;\
        t = sin_lat * cos_ha0;\
        y[i_out] = -l_ * sin_lat * sin_ha0 +\
                m_ * (cos_lat * cos_dec0 + t * sin_dec0) +\
                n_ * (cos_lat * sin_dec0 - t * cos_dec0);\
        t = cos_lat * cos_ha0;\
        z[i_out] = l_ * cos_lat * sin_ha0 +\
                m_ * (sin_lat * cos_dec0 - t * sin_dec0) +\
                n_ * (sin_lat * sin_dec0 + t * cos_dec0);\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
