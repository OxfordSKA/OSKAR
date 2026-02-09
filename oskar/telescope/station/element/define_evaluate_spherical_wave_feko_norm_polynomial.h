/* Copyright (c) 2025-2026, The OSKAR Developers. See LICENSE file. */

/*
 * Spherical wave evaluation method based on Matlab code by
 * David Davidson and Adrian Sutinjo.
 */

#define NP(m) workspace[(m) * num_points + i]

/* Returns the imaginary unit raised to an integer power 'n'. */
#define OSKAR_IPOW(NAME, FP, FP2) INLINE FP2 NAME(const int n)\
{\
    FP2 v;\
    switch (n & 3) /* (n % 4, but faster) */ {\
    case 0:\
        v.x = (FP) 1;\
        v.y = (FP) 0;\
        return v;\
    case 1:\
        v.x = (FP) 0;\
        v.y = (FP) 1;\
        return v;\
    case 2:\
        v.x = (FP) -1;\
        v.y = (FP) 0;\
        return v;\
    case 3:\
        v.x = (FP) 0;\
        v.y = (FP) -1;\
        return v;\
    }\
    v.x = v.y = (FP) 0;\
    return v;\
}\

#define OSKAR_EVALUATE_SPHERICAL_WAVE_SUM_FEKO_NORM(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const int n_max,\
        GLOBAL_IN(FP, root_n),\
        const int use_ticra_convention,\
        GLOBAL_IN(FP2, coeffs),\
        const FP coeff_scale,\
        GLOBAL_OUT(FP, workspace),\
        const int swap_xy,\
        const int offset,\
        GLOBAL_OUT(FP4c, pattern))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP2 e_theta[2], e_phi[2];\
    FP theta_ = theta[i];\
    const FP eta0 = (FP) 376.730313668; /* Post-2019 definition. */\
    const FP sqrt_fac = sqrt(eta0 / (2 * (FP) M_PI));\
    if (theta_ < (FP) 1e-5) theta_ = (FP) 1e-5; /* Avoid divide-by-zero. */\
    const FP phi_[2] = {phi_x[i], phi_y[i]};\
    e_theta[0].x = e_theta[0].y = e_phi[0].x = e_phi[0].y = (FP) 0;\
    e_theta[1].x = e_theta[1].y = e_phi[1].x = e_phi[1].y = (FP) 0;\
    if (phi_[0] != phi_[0]) {\
        /* Propagate NAN. */\
        e_theta[0].x = e_theta[0].y = e_phi[0].x = e_phi[0].y = phi_[0];\
        e_theta[1].x = e_theta[1].y = e_phi[1].x = e_phi[1].y = phi_[0];\
    }\
    else {\
        /* Compute cos and sin of direction coordinate for each antenna. */\
        FP cos_theta, sin_theta, cos_phi[2], sin_phi[2];\
        SINCOS(theta_, sin_theta, cos_theta);\
        SINCOS(phi_[0], sin_phi[0], cos_phi[0]);\
        SINCOS(phi_[1], sin_phi[1], cos_phi[1]);\
        /* Loop over degree of spherical wave. */\
        for (int n = 1; n <= n_max; ++n) {\
            /* Initialise cos(m phi), sin(m phi) for m = 0 at each antenna. */\
            FP cos_m_phi[2] = {1, 1};\
            FP sin_m_phi[2] = {0, 0};\
            M_CAT(legendre_norm_, FP)(\
                    n, num_points, i, cos_theta, sin_theta, root_n, workspace\
            );\
            NP(n + 1) = 0;\
            const FP sqrt_n = sqrt((FP) n * (n + 1));\
            const FP2 i_n = M_CAT(ipow_, FP)(n);\
            const FP2 i_np1 = M_CAT(ipow_, FP)(n + 1);\
            /* Loop over abs_m (0 to current n). */\
            for (int abs_m = 0; abs_m <= n; ++abs_m) {\
                /* Constants that don't depend on m (only abs_m). */\
                const FP cmn_constant = sqrt(\
                        (FP) (n + abs_m + 1) * (n - abs_m)\
                );\
                const FP np_1 = cmn_constant * NP(abs_m + 1);\
                const FP np_ds = NP(abs_m) / sin_theta;\
                const FP np_ds_abs_m = np_ds * abs_m * cos_theta;\
                const FP np_diff = np_ds_abs_m - np_1;\
                /* Do both +m and -m, but only do abs_m == 0 once! */\
                const int repeat_max = (abs_m == 0) ? 1 : 2;\
                for (int repeat = 0; repeat < repeat_max; ++repeat) {\
                    const int m = (repeat == 0) ? abs_m : -abs_m;\
                    const FP sign = (m >= 0) ? 1 : ((m & 1) ? (FP)-1 : (FP)1);\
                    const FP r = sign / sqrt_n;\
                    const FP np_ds_m = np_ds * m;\
                    const int m_indx = use_ticra_convention ? -m : m;\
                    const int j0 = 4 * (n * (n + 1) + m_indx - 1);\
                    /* Both antennas, X (a = 0) and Y (a = 1). */\
                    for (int a = 0; a < 2; ++a) {\
                        FP2 phi_const, te, tm, t1, t2;\
                        te = coeffs[j0 + 2 * a + 0];\
                        tm = coeffs[j0 + 2 * a + 1];\
                        te.x *= coeff_scale;\
                        te.y *= coeff_scale;\
                        tm.x *= coeff_scale;\
                        tm.y *= coeff_scale;\
                        if (use_ticra_convention) {\
                            te.y = -te.y;\
                            tm.y = -tm.y;\
                        }\
                        /* Construct e^(i m phi). */\
                        const FP sin_m_phi_m = (\
                                (m >= 0) ? sin_m_phi[a] : -sin_m_phi[a]\
                        );\
                        phi_const.x = cos_m_phi[a] * r;\
                        phi_const.y = sin_m_phi_m * r;\
                        /* Magic lines from Matlab code (simplified). */\
                        t1.x = -np_ds_m * te.x + tm.x * np_diff;\
                        t1.y = -np_ds_m * te.y + tm.y * np_diff;\
                        OSKAR_MUL_COMPLEX(t2, phi_const, i_n)\
                        OSKAR_MUL_ADD_COMPLEX(e_theta[a], t1, t2)\
                        t1.x = np_ds_m * tm.x - te.x * np_diff;\
                        t1.y = np_ds_m * tm.y - te.y * np_diff;\
                        OSKAR_MUL_COMPLEX(t2, phi_const, i_np1)\
                        OSKAR_MUL_ADD_COMPLEX(e_phi[a], t1, t2)\
                    }\
                }\
                /* Evaluate cos((m + 1) phi) and sin((m + 1) phi). */\
                for (int a = 0; a < 2; ++a) {\
                    const FP c = cos_m_phi[a];\
                    const FP s = sin_m_phi[a];\
                    cos_m_phi[a] = cos_phi[a] * c - sin_phi[a] * s;\
                    sin_m_phi[a] = sin_phi[a] * c + cos_phi[a] * s;\
                }\
            }\
        }\
    }\
    for (int a = 0; a < 2; ++a) {\
        e_theta[a].x *= sqrt_fac;\
        e_theta[a].y *= sqrt_fac;\
        e_phi[a].x *= sqrt_fac;\
        e_phi[a].y *= sqrt_fac;\
    }\
    const int i_out = i + offset;\
    if (swap_xy) {\
        pattern[i_out].a = e_theta[1];\
        pattern[i_out].b = e_phi[1];\
        pattern[i_out].c = e_theta[0];\
        pattern[i_out].d = e_phi[0];\
    }\
    else {\
        pattern[i_out].a = e_theta[0];\
        pattern[i_out].b = e_phi[0];\
        pattern[i_out].c = e_theta[1];\
        pattern[i_out].d = e_phi[1];\
    }\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
