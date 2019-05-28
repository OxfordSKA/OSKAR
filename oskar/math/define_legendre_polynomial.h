/* Copyright (c) 2019, The University of Oxford. See LICENSE file. */

/* See "Spherical Harmonic Lighting: The Gritty Details",
 * Robin Green GDC 2003. */

/* OUT is P_l^m(cos_theta) */
#define OSKAR_LEGENDRE1(FP, L, M, COS_THETA, SIN_THETA, OUT) {\
    FP p0_ = (FP) 1;\
    if (M > 0) {\
        FP fact = (FP) 1;\
        for (int i_ = 1; i_ <= M; ++i_) {\
            p0_ *= (-fact) * SIN_THETA;\
            fact += (FP) 2;\
        }\
    }\
    if (L == M) OUT = p0_;\
    else {\
        FP p1_ = COS_THETA * (2 * M + 1) * p0_;\
        if (L == M + 1) OUT = p1_;\
        else {\
            OUT = (FP)0;\
            for (int i_ = M + 2; i_ <= L; ++i_) {\
                OUT = ((2 * i_ - 1) * COS_THETA * p1_ - (i_ + M - 1) * p0_) / (i_ - M);\
                p0_ = p1_; p1_ = OUT;\
            }\
        }\
    }\
    }\


/* OUT0 is P_l^m(cos_theta),
 * OUT1 is P_l^m(cos_theta) / sin_theta,
 * OUT2 is d/d(cos_theta){P_l^m(cos_theta)} * sin_theta. */
#define OSKAR_LEGENDRE2(FP, L, M, COS_THETA, SIN_THETA, OUT0, OUT1, OUT2) {\
    FP p0_ = (FP) 1, p1_;\
    if (M > 0) {\
        FP fact = (FP) 1;\
        for (int i_ = 1; i_ <= M; ++i_) {\
            p0_ *= (-fact) * SIN_THETA;\
            fact += (FP) 2;\
        }\
    }\
    OUT0 = COS_THETA * (2 * M + 1) * p0_;\
    if (L == M) {\
        p1_ = OUT0;\
        OUT0 = p0_;\
    }\
    else {\
        p1_ = OUT0;\
        for (int i_ = M + 2; i_ <= L + 1; ++i_) {\
            OUT0 = p1_;\
            p1_ = ((2 * i_ - 1) * COS_THETA * OUT0 - (i_ + M - 1) * p0_) / (i_ - M);\
            p0_ = OUT0;\
        }\
    }\
    if (SIN_THETA != (FP) 0) {\
        /* BOTH of these are divides. */\
        OUT1 = OUT0 / SIN_THETA;\
        OUT2 = (COS_THETA * OUT0 * (L + 1) - p1_ * (L - M + 1)) / SIN_THETA;\
    }\
    else {\
        OUT1 = OUT2 = (FP) 0;\
    }\
    }\

