/* Copyright (c) 2025-2026, The OSKAR Developers. See LICENSE file. */

#define OLPN(m) y[(m) * num_points + idx_point]

#define OSKAR_LEGENDRE_NORM(NAME, FP) DEVICE_FUNC void NAME (\
        const int n,\
        const int num_points,\
        const int idx_point,\
        const FP cos_theta,\
        const FP sin_theta,\
        GLOBAL_IN(FP, root_n),\
        FP* y)\
{\
    const FP tol = sqrt((FP) 1.17549435082228750796873653722224568e-38);\
    const FP root_n_plus_half = sqrt(n + (FP) 0.5);\
    const FP twocot = sin_theta > (FP) 0 ? -2 * cos_theta / sin_theta : (FP) 0;\
    const FP sn = sin_theta == (FP) 0 ? (FP) 0 : (\
            (n % 2 == 0) ? pow(sin_theta, n) : -pow(sin_theta, n)\
    );\
    for (int m = 0; m <= n; ++m) OLPN(m) = (FP) 0; /* Clear workspace. */\
    if (sin_theta > 0 && fabs(sn) <= tol) {\
        /* Underflow branch. */\
        const FP tstart = (FP) 1.1920928955078125e-7; /* FLT_EPSILON */\
        const FP v = (FP) 9.2 - log(tol) / (n * sin_theta);\
        const FP w = (FP) 1 / log(v);\
        const FP m1d = (FP) 1 + n * sin_theta * v * w * (\
                (FP) 1.0058 + w * ((FP) 3.819 - w * (FP) 12.173)\
        );\
        int m1 = (int) floor(m1d);\
        if (m1 < 1) m1 = 1;\
        if (m1 > n) m1 = n;\
        const FP sgn_m1 = ((m1 % 2) == 1) ? (FP) 1 : (FP) -1;\
        const FP sgn_n1 = (((n + 1) % 2) == 1) ? (FP) 1 : (FP) -1;\
        OLPN(m1 - 1) = (cos_theta < 0) ? (sgn_n1 * tstart) : (sgn_m1 * tstart);\
        FP sumsq = tol;\
        for (int m = m1 - 2; m >= 0; --m) {\
            const FP a = (m + 1) * twocot * OLPN(m + 1);\
            const FP b = root_n[n + m + 2] * root_n[n - m - 1] * OLPN(m + 2);\
            OLPN(m) = (a - b) / (root_n[n + m + 1] * root_n[n - m]);\
            sumsq += OLPN(m) * OLPN(m);\
        }\
        const FP scale = (FP) 1 / sqrt(2 * sumsq - OLPN(0) * OLPN(0));\
        for (int r = 0; r <= m1; ++r) {\
            OLPN(r) *= scale;\
        }\
    }\
    if (cos_theta != (FP) 1 && fabs(sn) >= tol) {\
        /* Non-underflow branch. */\
        FP c = (FP) 1;\
        for (int k = 1; k <= n; ++k) {\
            c *= ( (FP) 1 - ((FP) 1 / ( (FP) 2 * k )) );\
        }\
        OLPN(n) = sqrt(c) * sn;\
        OLPN(n - 1) = OLPN(n) * twocot * (FP) n / root_n[2 * n];\
        for (int m = n - 2; m >= 0; --m) {\
            const FP a = (m + 1) * twocot * OLPN(m + 1);\
            const FP b = root_n[n + m + 2] * root_n[n - m - 1] * OLPN(m + 2);\
            OLPN(m) = (a - b) / (root_n[n + m + 1] * root_n[n - m]);\
        }\
    }\
    if (sin_theta == (FP) 0) OLPN(0) = pow(cos_theta, n); /* Polar case. */\
    for (int m = 0; m <= n; ++m) {\
        /* Apply normalisation. */\
        const FP sgn = ((m % 2) == 0) ? (FP) 1 : (FP) -1;\
        OLPN(m) *= root_n_plus_half * sgn;\
    }\
}
