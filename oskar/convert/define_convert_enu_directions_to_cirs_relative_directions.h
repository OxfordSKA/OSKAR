/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_ENU_DIR_TO_CIRS_REL_DIR(NAME, FP) KERNEL(NAME) (\
        const int offset_in, const int num, GLOBAL_IN(FP, x),\
        GLOBAL_IN(FP, y), GLOBAL_IN(FP, z),\
        const FP sin_ha0, const FP cos_ha0, const FP sin_dec0,\
        const FP cos_dec0, const FP sin_lat, const FP cos_lat,\
        const FP local_pm_x, const FP local_pm_y, const FP diurnal_aberration,\
        const int offset_out, GLOBAL_OUT(FP, l), GLOBAL_OUT(FP, m),\
        GLOBAL_OUT(FP, n))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const int i_in = i + offset_in, i_out = i + offset_out;\
    FP x_ = x[i_in], y_ = y[i_in], z_ = z[i_in], x2, y2, z2;\
    /* ENU directions to Cartesian -HA,Dec. */\
    /* This is the first (latitude) stage of the original transformation:
     *   rotate by -phi about x.
     * Note axes are permuted so that x -> Y, y -> Z and z -> X.
     * X towards local meridian, Z towards NCP, Y towards local east. */\
    y2 = x_;\
    z2 = y_ * cos_lat + z_ * sin_lat;\
    x2 = -y_ * sin_lat + z_ * cos_lat;\
    /* Diurnal aberration. */\
    const FP f = (FP)1 + diurnal_aberration * y2;\
    x_ = f * x2;\
    y_ = f * (y2 - diurnal_aberration);\
    z_ = f * z2;\
    /* Polar motion, permuting axes back to ENU directions. */\
    const FP w = local_pm_x * x_ - local_pm_y * y_ + z_;\
    z2 = x_ - local_pm_x * w;\
    x2 = y_ + local_pm_y * w;\
    y2 = w - (local_pm_x * local_pm_x + local_pm_y * local_pm_y) * z_;\
    /* ENU directions to CIRS relative directions. */\
    /* This is the final two stages of the original transformation:
     *   rotate by ha0 around y, then by dec0 around x. */\
    const FP t1 = x2 * sin_ha0;\
    const FP t2 = z2 * cos_ha0;\
    l[i_out] =  x2 * cos_ha0 + z2 * sin_ha0;\
    m[i_out] =  t1 * sin_dec0 + y2 * cos_dec0 - t2 * sin_dec0;\
    n[i_out] = -t1 * cos_dec0 + y2 * sin_dec0 + t2 * cos_dec0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
