/* Copyright (c) 2019-2020, The University of Oxford. See LICENSE file. */

/* Spherical wave evaluation method based on Matlab code by
 * Christophe Craeye, Quentin Gueuning and Eloy de Lera Acedo.
 * See "Spherical near-field antenna measurements", J. E. Hansen, 1988 */

#define OSKAR_SPH_WAVE(FP2, M, A_TE, A_TM, C_THETA, C_PHI) {\
    FP2 qq, dd;\
    qq.x = -cos_p * dpms;\
    qq.y = -sin_p * dpms;\
    dd.x = -sin_p * pds * (M);\
    dd.y = cos_p * pds * (M);\
    OSKAR_MUL_ADD_COMPLEX(C_PHI, qq, A_TM)\
    OSKAR_MUL_SUB_COMPLEX(C_PHI, dd, A_TE)\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, dd, A_TM)\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, qq, A_TE)\
    }\

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const int l_max,\
        GLOBAL_IN(FP4c, alpha),\
        const int offset,\
        GLOBAL_OUT(FP4c, pattern))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 Xp, Xt, Yp, Yt;\
    FP theta_;\
    MAKE_ZERO2(FP, Xp); MAKE_ZERO2(FP, Xt);\
    MAKE_ZERO2(FP, Yp); MAKE_ZERO2(FP, Yt);\
    theta_ = theta[i];\
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
            const FP f_ = (2 * l + 1) / (4 * ((FP)M_PI) * l * (l + 1));\
            for (int abs_m = l; abs_m >=0; --abs_m) {\
                FP p, pds, dpms, sin_p, cos_p;\
                OSKAR_LEGENDRE2(FP, l, abs_m, cos_t, sin_t, p, pds, dpms)\
                if (abs_m == 0) {\
                    sin_p = (FP)0; cos_p = sqrt(f_);\
                    const FP4c alpha_ = alpha[ind0];\
                    OSKAR_SPH_WAVE(FP2, 0, alpha_.a, alpha_.b, Xt, Xp)\
                    OSKAR_SPH_WAVE(FP2, 0, alpha_.c, alpha_.d, Yt, Yp)\
                }\
                else {\
                    FP d_fact = (FP)1, s_fact = (FP)1;\
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
                    OSKAR_SPH_WAVE(FP2, -abs_m, alpha_m.a, alpha_m.b, Xt, Xp)\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE(FP2,  abs_m, alpha_p.a, alpha_p.b, Xt, Xp)\
                    p = -abs_m * phi_y_;\
                    SINCOS(p, sin_p, cos_p);\
                    sin_p *= nf; cos_p *= nf;\
                    OSKAR_SPH_WAVE(FP2, -abs_m, alpha_m.c, alpha_m.d, Yt, Yp)\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE(FP2,  abs_m, alpha_p.c, alpha_p.d, Yt, Yp)\
                }\
            }\
        }\
    }\
    /* For some reason the theta/phi components must be reversed? */\
    pattern[i + offset].a = Xp;\
    pattern[i + offset].b = Xt;\
    pattern[i + offset].c = Yp;\
    pattern[i + offset].d = Yt;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
