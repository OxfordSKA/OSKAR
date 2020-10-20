/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_UPDATE_HORIZON_MASK(NAME, FP) KERNEL(NAME) (const int num,\
        GLOBAL_IN(FP, l), GLOBAL_IN(FP, m), GLOBAL_IN(FP, n),\
        const FP l_mul, const FP m_mul, const FP n_mul,\
        GLOBAL_OUT(int, mask))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    mask[i] |= ((l[i] * l_mul + m[i] * m_mul + n[i] * n_mul) > (FP) 0);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_UPDATE_HORIZON_MASK2(NAME, FP) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, ra_rad), GLOBAL_IN(FP, dec_rad),\
        const FP lst_rad, const FP cos_lat, const FP sin_lat,\
        GLOBAL_OUT(int, mask))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_dec, cos_dec;\
    const FP cos_ha = cos(lst_rad - ra_rad[i]);\
    SINCOS(dec_rad[i], sin_dec, cos_dec);\
    mask[i] |= ((sin_lat * sin_dec + cos_lat * cos_dec * cos_ha) > (FP) 0);\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
