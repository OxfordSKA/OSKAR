/* Copyright (c) 2019-2025, The OSKAR Developers. See LICENSE file. */

/* Spherical wave evaluation method based on Matlab code by
 * Christophe Craeye, Quentin Gueuning and Eloy de Lera Acedo.
 *
 * Modified for Galileo conventions by Maciej Serylak.
 *
 * See "Spherical near-field antenna measurements", J. E. Hansen, 1988
 */

#define OSKAR_SPH_WAVE3(FP2, M, A_TE, A_TM, C_THETA, C_PHI) {\
    FP2 qq, dd, qq_te, qq_tm, dd_te, dd_tm;\
    qq.x = cos_p * dpms;\
    qq.y = sin_p * dpms;\
    dd.x = -sin_p * pds * (M);\
    dd.y = cos_p * pds * (M);\
    OSKAR_MUL_COMPLEX(qq_te, qq, flip_te);\
    OSKAR_MUL_COMPLEX(qq_tm, qq, flip_tm);\
    OSKAR_MUL_COMPLEX(dd_te, dd, flip_te);\
    OSKAR_MUL_COMPLEX(dd_tm, dd, flip_tm);\
    qq_te.x *= bh_factor;\
    qq_te.y *= bh_factor;\
    qq_tm.x *= bh_factor;\
    qq_tm.y *= bh_factor;\
    dd_te.x *= bh_factor;\
    dd_te.y *= bh_factor;\
    dd_tm.x *= bh_factor;\
    dd_tm.y *= bh_factor;\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, dd_te, A_TE)\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, qq_tm, A_TM)\
    OSKAR_MUL_ADD_COMPLEX(C_PHI, dd_tm, A_TM)\
    OSKAR_MUL_SUB_COMPLEX(C_PHI, qq_te, A_TE)\
    }\

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_GALILEO(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const int l_max,\
        GLOBAL_IN(FP4c, alpha),\
        const int swap_xy,\
        const int offset,\
        GLOBAL_OUT(FP4c, pattern))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 Xp, Xt, Yp, Yt;\
    FP theta_;\
    MAKE_ZERO2(FP, Xp); MAKE_ZERO2(FP, Xt);\
    MAKE_ZERO2(FP, Yp); MAKE_ZERO2(FP, Yt);\
    theta_ = theta[i];\
    /* Intrinsic impedance of free space in order to obtain antenna voltage scaling. */\
    const FP Zo = 376.730313668; /* Post-2019 definition. */\
    /* Hack to avoid divide-by-zero (also in Matlab code!). */\
    if (theta_ < (FP)1e-5) theta_ = (FP)1e-5;\
    const FP phi_x_ = phi_x[i];\
    const FP phi_y_ = phi_y[i];\
    /* Propagate NAN. */\
    if (phi_x_ != phi_x_) {\
        Xp.x = Xp.y = Xt.x = Xt.y = phi_x_;\
        Yp.x = Yp.y = Yt.x = Yt.y = phi_x_;\
    }\
    else {\
        FP sin_t, cos_t;\
        SINCOS(theta_, sin_t, cos_t);\
        for (int l = 1; l <= l_max; ++l) {\
            const int ind0 = l * l - 1 + l;\
            const FP f_ = Zo * (2 * l + 1) / (4 * ((FP)M_PI) * l * (l + 1));\
            FP2 flip_tm, flip_te;\
            MAKE_ZERO2(FP, flip_te);\
            MAKE_ZERO2(FP, flip_tm);\
            FP remainder = l % 4;\
            if (remainder == 0) {\
                flip_tm.x = 1;\
                flip_tm.y = 0;\
                flip_te.x = 0;\
                flip_te.y = -1;\
            } else if (remainder == 1) {\
                flip_tm.x = 0;\
                flip_tm.y = -1;\
                flip_te.x = -1;\
                flip_te.y = 0;\
            } else if (remainder == 2) {\
                flip_tm.x = -1;\
                flip_tm.y = 0;\
                flip_te.x = 0;\
                flip_te.y = 1;\
            } else if (remainder == 3) {\
                flip_tm.x = 0;\
                flip_tm.y = 1;\
                flip_te.x = 1;\
                flip_te.y = 0;\
            }\
            for (int abs_m = l; abs_m >=0; --abs_m) {\
                FP p, pds, dpms, sin_p, cos_p;\
                OSKAR_LEGENDRE3(FP, l, abs_m, cos_t, sin_t, p, pds, dpms)\
                if (abs_m == 0) {\
                    FP bh_factor = (FP)1;\
                    sin_p = (FP)0; cos_p = sqrt(f_);\
                    const FP4c alpha_ = alpha[ind0];\
                    OSKAR_SPH_WAVE3(FP2, 0, alpha_.a, alpha_.b, Xt, Xp)\
                    OSKAR_SPH_WAVE3(FP2, 0, alpha_.c, alpha_.d, Yt, Yp)\
                }\
                else {\
                    FP d_fact = (FP)1, s_fact = (FP)1, bh_factor = (FP)1;\
                    const int d_ = l - abs_m, s_ = l + abs_m;\
                    for (int i_ = 2; i_ <= d_; ++i_) d_fact *= i_;\
                    for (int i_ = 2; i_ <= s_; ++i_) s_fact *= i_;\
                    const FP ff = f_ * d_fact / s_fact;\
                    const FP nf = sqrt(ff);\
                    const FP4c alpha_m = alpha[ind0 - abs_m];\
                    const FP4c alpha_p = alpha[ind0 + abs_m];\
                    p = -abs_m * phi_x_;\
                    SINCOS(p, sin_p, cos_p);\
                    sin_p *= nf; cos_p *= nf;\
                    OSKAR_SPH_WAVE3(FP2, -abs_m, alpha_m.a, alpha_m.b, Xt, Xp)\
                    bh_factor = pow((FP)-1, (FP)abs_m);\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE3(FP2,  abs_m, alpha_p.a, alpha_p.b, Xt, Xp)\
                    bh_factor = (FP)1;\
                    p = -abs_m * phi_y_;\
                    SINCOS(p, sin_p, cos_p);\
                    sin_p *= nf; cos_p *= nf;\
                    OSKAR_SPH_WAVE3(FP2, -abs_m, alpha_m.c, alpha_m.d, Yt, Yp)\
                    sin_p = -sin_p;\
                    bh_factor = pow((FP)-1, (FP)abs_m);\
                    OSKAR_SPH_WAVE3(FP2,  abs_m, alpha_p.c, alpha_p.d, Yt, Yp)\
                }\
            }\
        }\
    }\
    Xt.y *= -1.0;\
    Xp.y *= -1.0;\
    Yt.y *= -1.0;\
    Yp.y *= -1.0;\
    if (swap_xy) {\
        pattern[i + offset].a = Yt;\
        pattern[i + offset].b = Yp;\
        pattern[i + offset].c = Xt;\
        pattern[i + offset].d = Xp;\
    }\
    else {\
        pattern[i + offset].a = Xt;\
        pattern[i + offset].b = Xp;\
        pattern[i + offset].c = Yt;\
        pattern[i + offset].d = Yp;\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
