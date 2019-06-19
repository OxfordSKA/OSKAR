/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

/* Spherical wave evaluation method based on Matlab code by
 * Christophe Craeye, Quentin Gueuning and Eloy de Lera Acedo.
 * See "Spherical near-field antenna measurements", J. E. Hansen, 1988 */

#define OSKAR_SPH_WAVE(FP2, M, A_TE, A_TM, C_THETA, C_PHI) {\
    FP2 qq, dd;\
    qq.x = -cos_p * dpms;\
    qq.y = -sin_p * dpms;\
    dd.x = -sin_p * pds * (M);\
    dd.y = cos_p * pds * (M);\
    const FP2 te_ = A_TE[ind0 + (M)];\
    const FP2 tm_ = A_TM[ind0 + (M)];\
    OSKAR_MUL_ADD_COMPLEX(C_PHI, qq, tm_)\
    OSKAR_MUL_SUB_COMPLEX(C_PHI, dd, te_)\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, dd, tm_)\
    OSKAR_MUL_ADD_COMPLEX(C_THETA, qq, te_)\
    }\

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM(NAME, FP, FP2)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int l_max,\
        GLOBAL_IN(FP2, a_te),\
        GLOBAL_IN(FP2, a_tm),\
        const int stride,\
        const int E_theta_offset,\
        const int E_phi_offset,\
        GLOBAL FP2* E_theta,\
        GLOBAL FP2* E_phi)\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 Ep, Et; /* Originally gvv, ghh. */\
    MAKE_ZERO2(FP, Ep); MAKE_ZERO2(FP, Et);\
    const int i_out = i * stride;\
    const int theta_out = i_out + E_theta_offset;\
    const int phi_out   = i_out + E_phi_offset;\
    const FP phi_ = phi[i], theta_ = theta[i];\
    /* Propagate NAN. */\
    if (phi_ != phi_) {\
        Ep.x = Ep.y = Et.x = Et.y = phi_;\
    }\
    else {\
        FP sin_t, cos_t;\
        SINCOS(theta_, sin_t, cos_t);\
        for (int l = 1; l <= l_max; ++l) {\
            const int ind0 = (2 * l_max + 1) * (l - 1) + l;\
            const FP f_ = (2 * l + 1) / (4 * (FP)M_PI * l * (l + 1));\
            for (int abs_m = l; abs_m >=0; --abs_m) {\
                FP p, pds, dpms, sin_p, cos_p;\
                OSKAR_LEGENDRE2(FP, l, abs_m, cos_t, sin_t, p, pds, dpms)\
                if (abs_m == 0) {\
                    sin_p = (FP)0; cos_p = sqrt(f_);\
                    OSKAR_SPH_WAVE(FP2, 0, a_te, a_tm, Et, Ep)\
                }\
                else {\
                    FP d_fact = (FP)1, s_fact = (FP)1;\
                    const int d_ = l - abs_m, s_ = l + abs_m;\
                    for (int i_ = 2; i_ <= d_; ++i_) d_fact *= i_;\
                    for (int i_ = 2; i_ <= s_; ++i_) s_fact *= i_;\
                    const FP ff = f_ * d_fact / s_fact;\
                    const FP nf = sqrt(ff);\
                    p = -abs_m * phi_;\
                    SINCOS(p, sin_p, cos_p);\
                    sin_p *= nf; cos_p *= nf;\
                    OSKAR_SPH_WAVE(FP2, -abs_m, a_te, a_tm, Et, Ep)\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE(FP2,  abs_m, a_te, a_tm, Et, Ep)\
                }\
            }\
        }\
    }\
    E_theta[theta_out] = Et;\
    E_phi[phi_out] = Ep;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM2(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int l_max,\
        GLOBAL_IN(FP2, x_te),\
        GLOBAL_IN(FP2, x_tm),\
        GLOBAL_IN(FP2, y_te),\
        GLOBAL_IN(FP2, y_tm),\
        const int stride,\
        const int offset,\
        GLOBAL_OUT(FP4c, pattern))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 Xp, Xt, Yp, Yt;\
    MAKE_ZERO2(FP, Xp); MAKE_ZERO2(FP, Xt);\
    MAKE_ZERO2(FP, Yp); MAKE_ZERO2(FP, Yt);\
    const int i_out = i * stride + offset;\
    const FP phi_ = phi[i], theta_ = theta[i];\
    /* Propagate NAN. */\
    if (phi_ != phi_) {\
        Xp.x = Xp.y = Xt.x = Xt.y = phi_;\
        Yp.x = Yp.y = Yt.x = Yt.y = phi_;\
    }\
    else {\
        FP sin_t, cos_t;\
        SINCOS(theta_, sin_t, cos_t);\
        for (int l = 1; l <= l_max; ++l) {\
            const int ind0 = (2 * l_max + 1) * (l - 1) + l;\
            const FP f_ = (2 * l + 1) / (4 * (FP)M_PI * l * (l + 1));\
            for (int abs_m = l; abs_m >=0; --abs_m) {\
                FP p, pds, dpms, sin_p, cos_p;\
                OSKAR_LEGENDRE2(FP, l, abs_m, cos_t, sin_t, p, pds, dpms)\
                if (abs_m == 0) {\
                    sin_p = (FP)0; cos_p = sqrt(f_);\
                    OSKAR_SPH_WAVE(FP2, 0, x_te, x_tm, Xt, Xp)\
                    OSKAR_SPH_WAVE(FP2, 0, y_te, y_tm, Yt, Yp)\
                }\
                else {\
                    FP temp, d_fact = (FP)1, s_fact = (FP)1;\
                    const int d_ = l - abs_m, s_ = l + abs_m;\
                    for (int i_ = 2; i_ <= d_; ++i_) d_fact *= i_;\
                    for (int i_ = 2; i_ <= s_; ++i_) s_fact *= i_;\
                    const FP ff = f_ * d_fact / s_fact;\
                    const FP nf = sqrt(ff);\
                    p = -abs_m * phi_;\
                    SINCOS(p, sin_p, cos_p);\
                    sin_p *= nf; cos_p *= nf;\
                    OSKAR_SPH_WAVE(FP2, -abs_m, x_te, x_tm, Xt, Xp)\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE(FP2,  abs_m, x_te, x_tm, Xt, Xp)\
                    temp = sin_p;\
                    sin_p = cos_p;\
                    cos_p = temp; /* Already negated. */\
                    OSKAR_SPH_WAVE(FP2, -abs_m, y_te, y_tm, Yt, Yp)\
                    sin_p = -sin_p;\
                    OSKAR_SPH_WAVE(FP2,  abs_m, y_te, y_tm, Yt, Yp)\
                }\
            }\
        }\
    }\
    pattern[i_out].a = Xt;\
    pattern[i_out].b = Xp;\
    pattern[i_out].c = Yt;\
    pattern[i_out].d = Yp;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
