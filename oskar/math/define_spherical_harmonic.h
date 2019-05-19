/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

/* See "Spherical Harmonic Lighting: The Gritty Details",
 * Robin Green, GDC, 2003. */

#define OSKAR_SPHERICAL_HARMONIC_SUM_REAL(NAME, FP)\
KERNEL(NAME) (\
        const int num_points,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int l_max,\
        GLOBAL_IN(FP, coeff),\
        const int stride,\
        const int offset_out,\
        GLOBAL_OUT(FP, surface))\
{\
    KERNEL_LOOP_PAR_X(int, i, 0, num_points)\
    FP sum = (FP)0;\
    const FP phi_ = phi[i], theta_ = theta[i];\
    if (phi_ != phi_) sum = phi_; /* Propagate NAN. */\
    else {\
        FP sin_t, cos_t;\
        SINCOS(theta_, sin_t, cos_t);\
        for (int l = 0; l <= l_max; ++l) {\
            FP val;\
            const int ind0 = l * l + l;\
            const FP f_ = (2 * l + 1) / (4 * (FP)M_PI);\
            const FP c0 = coeff[ind0];\
            if (c0 != (FP)0) {\
                OSKAR_LEGENDRE1(FP, l, 0, cos_t, sin_t, val)\
                sum += c0 * val * sqrt(f_);\
            }\
            for (int abs_m = 1; abs_m <= l; ++abs_m) {\
                FP sin_p, cos_p, d_fact = (FP)1, s_fact = (FP)1;\
                const FP cmm = coeff[ind0 - abs_m], cpm = coeff[ind0 + abs_m];\
                if ((cpm == (FP)0) && (cmm == (FP)0)) continue;\
                OSKAR_LEGENDRE1(FP, l, abs_m, cos_t, sin_t, val)\
                const int d_ = l - abs_m, s_ = l + abs_m;\
                for (int i_ = 2; i_ <= d_; ++i_) d_fact *= i_;\
                for (int i_ = 2; i_ <= s_; ++i_) s_fact *= i_;\
                val *= sqrt(2 * f_ * d_fact / s_fact);\
                const FP p = abs_m * phi_;\
                SINCOS(p, sin_p, cos_p);\
                sum += (val * (cpm * cos_p + cmm * sin_p));\
            }\
        }\
    }\
    surface[i * stride + offset_out] = sum;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
