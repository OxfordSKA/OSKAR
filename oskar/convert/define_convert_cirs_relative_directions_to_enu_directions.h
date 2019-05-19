/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_CIRS_REL_DIR_TO_ENU_DIR(NAME, FP) KERNEL(NAME) (\
        const int at_origin, const int bypass, const int offset_in,\
        const int num, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m), GLOBAL_IN(FP, n),\
        const FP sin_ha0, const FP cos_ha0, const FP sin_dec0,\
        const FP cos_dec0, const FP sin_lat, const FP cos_lat,\
        const FP local_pm_x, const FP local_pm_y, const FP diurnal_aberration,\
        const int offset_out, GLOBAL_OUT(FP, x), GLOBAL_OUT(FP, y),\
        GLOBAL_OUT(FP, z))\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    const int i_in = i + offset_in, i_out = i + offset_out;\
    FP l_, m_, n_, x2, y2, z2;\
    if (at_origin) {\
        l_ = (FP)0; m_ = (FP)0; n_ = (FP)1;\
    } else {\
        l_ = l[i_in]; m_ = m[i_in]; n_ = n[i_in];\
    }\
    if (bypass) {\
        x[i_out] = l_; y[i_out] = m_; z[i_out] = n_;\
    } else {\
        /* CIRS relative directions to Cartesian -HA, Dec. */\
        /* This is the first two stages of original transformation:
         *   rotate by -dec0 about x, then rotate by -ha0 about y.
         * Note axes are permuted so that x -> Y, y -> Z and z -> X.
         * X towards local meridian, Z towards NCP, Y towards local east. */\
        const FP t1 = m_ * sin_dec0;\
        const FP t2 = n_ * cos_dec0;\
        y2 = l_ * cos_ha0  + t1 * sin_ha0 - t2 * sin_ha0;\
        z2 = m_ * cos_dec0 + n_ * sin_dec0;\
        x2 = l_ * sin_ha0  - t1 * cos_ha0 + t2 * cos_ha0;\
        /* Polar motion. */\
        l_ = x2 + local_pm_x * z2;\
        m_ = y2 - local_pm_y * z2;\
        n_ = z2 - local_pm_x * x2 + local_pm_y * y2;\
        /* Diurnal aberration. */\
        const FP f = (FP)1 - diurnal_aberration * m_;\
        x2 = f * l_;\
        y2 = f * (m_ + diurnal_aberration);\
        z2 = f * n_;\
        /* Cartesian -HA, Dec to Cartesian ENU directions. */\
        /* This is the final (latitude) stage of original transformation. */\
        x[i_out] = y2;\
        y[i_out] = -(sin_lat * x2 - cos_lat * z2);\
        z[i_out] = cos_lat * x2 + sin_lat * z2;\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
