/* Copyright (c) 2011-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_JONES_R(NAME, FP, FP4c) KERNEL(NAME) (\
        const int        num_sources,\
        GLOBAL_IN(FP,    ra_rad),\
        GLOBAL_IN(FP,    dec_rad),\
        const FP         cos_lat,\
        const FP         sin_lat,\
        const FP         lst_rad,\
        const int        offset_out,\
        GLOBAL_OUT(FP4c, jones))\
{\
    KERNEL_LOOP_X(int, i, 0, num_sources)\
    FP cos_ha, sin_ha, cos_dec, sin_dec, cos_par_ang, sin_par_ang;\
    FP4c J;\
    const FP ha = lst_rad - ra_rad[i];\
    const FP dec = dec_rad[i];\
    SINCOS(ha, sin_ha, cos_ha);\
    SINCOS(dec, sin_dec, cos_dec);\
    const FP y = cos_lat * sin_ha;\
    const FP x = sin_lat * cos_dec - cos_lat * sin_dec * cos_ha;\
    const FP par_ang = atan2(y, x);\
    SINCOS(par_ang, sin_par_ang, cos_par_ang);\
    J.a.x = cos_par_ang; J.b.x = -sin_par_ang;\
    J.c.x = sin_par_ang; J.d.x = cos_par_ang;\
    MAKE_ZERO(FP, J.a.y = J.b.y = J.c.y = J.d.y);\
    jones[i + offset_out] = J;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
