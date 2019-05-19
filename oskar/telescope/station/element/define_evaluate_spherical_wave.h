/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

/* Spherical wave evaluation method based on Matlab code by
 * Christophe Craeye, Quentin Gueuning and Eloy de Lera Acedo.
 * See "Spherical near-field antenna measurements", J. E. Hansen, 1988 */

#define OSKAR_SPHERICAL_WAVE_M(M, PHI) {\
    p = (M) * (PHI);\
    SINCOS(p, sin_p, cos_p);\
    sin_p *= nf; cos_p *= nf;\
    qq.x = -cos_p * dpms;\
    qq.y = -sin_p * dpms;\
    dd.x = -sin_p * pds * (M);\
    dd.y = cos_p * pds * (M);\
    a_te = alpha_te[ind0 + (M)];\
    a_tm = alpha_tm[ind0 + (M)];\
    OSKAR_MUL_ADD_COMPLEX(comp_phi, qq, a_tm)\
    OSKAR_MUL_SUB_COMPLEX(comp_phi, dd, a_te)\
    OSKAR_MUL_ADD_COMPLEX(comp_theta, dd, a_tm)\
    OSKAR_MUL_ADD_COMPLEX(comp_theta, qq, a_te)\
    }\

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM(NAME, FP, FP2)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int l_max,\
        GLOBAL_IN(FP2, alpha_te),\
        GLOBAL_IN(FP2, alpha_tm),\
        const int stride,\
        const int E_theta_offset,\
        const int E_phi_offset,\
        GLOBAL FP2* E_theta,\
        GLOBAL FP2* E_phi)\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 comp_phi, comp_theta; /* Originally gvv, ghh. */\
    MAKE_ZERO2(FP, comp_phi); MAKE_ZERO2(FP, comp_theta);\
    const int i_out = i * stride;\
    const int theta_out = i_out + E_theta_offset;\
    const int phi_out   = i_out + E_phi_offset;\
    const FP phi_ = phi[i], theta_ = theta[i];\
    /* Propagate NAN. */\
    if (phi_ != phi_) {\
        comp_phi.x = comp_phi.y = comp_theta.x = comp_theta.y = phi_;\
    }\
    else {\
        FP sin_t, cos_t;\
        SINCOS(theta_, sin_t, cos_t);\
        for (int l = 1; l <= l_max; ++l) {\
            const int ind0 = (2 * l_max + 1) * (l - 1) + l;\
            const FP f_ = (2 * l + 1) / (4 * (FP)M_PI * l * (l + 1));\
            for (int abs_m = l; abs_m >=0; --abs_m) {\
                FP2 qq, dd, a_te, a_tm;\
                FP nf, p, pds, dpms, sin_p, cos_p;\
                OSKAR_LEGENDRE2(FP, l, abs_m, cos_t, sin_t, p, pds, dpms)\
                if (abs_m == 0) {\
                    nf = sqrt(f_);\
                    OSKAR_SPHERICAL_WAVE_M(0, phi_)\
                }\
                else {\
                    FP d_fact = (FP)1, s_fact = (FP)1;\
                    const int d_ = l - abs_m, s_ = l + abs_m;\
                    for (int i_ = 2; i_ <= d_; ++i_) d_fact *= i_;\
                    for (int i_ = 2; i_ <= s_; ++i_) s_fact *= i_;\
                    nf = sqrt(f_ * d_fact / s_fact);\
                    OSKAR_SPHERICAL_WAVE_M(-abs_m, phi_)\
                    OSKAR_SPHERICAL_WAVE_M(abs_m, phi_)\
                }\
            }\
        }\
    }\
    E_theta[theta_out] = comp_theta;\
    E_phi[phi_out] = comp_phi;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
