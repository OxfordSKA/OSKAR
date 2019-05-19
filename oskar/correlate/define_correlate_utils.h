/* Copyright (c) 2013-2019, The University of Oxford. See LICENSE file. */

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */

#define OSKAR_BASELINE_TERMS(FP, S_UP, S_UQ, S_VP, S_VQ, S_WP, S_WQ, UU, VV, WW, UU2, VV2, UUVV, UV_LEN) {\
        UU   = (S_UP - S_UQ) * inv_wavelength;\
        VV   = (S_VP - S_VQ) * inv_wavelength;\
        WW   = (S_WP - S_WQ) * inv_wavelength;\
        UU2  = UU * UU; VV2  = VV * VV;\
        UV_LEN = sqrt((FP) (UU2 + VV2));\
        UUVV = 2 * UU * VV;\
        const FP f = ((FP) M_PI) * frac_bandwidth;\
        UU *= f; VV *= f; WW *= f;}\

#define OSKAR_BASELINE_DELTAS(FP, S_XP, S_XQ, S_YP, S_YQ, DU, DV, DW) {\
        FP xx, yy, rot_angle, temp, sin_HA, cos_HA, sin_Dec, cos_Dec;\
        SINCOS(gha0_rad, sin_HA, cos_HA);\
        SINCOS(dec0_rad, sin_Dec, cos_Dec);\
        temp = ((FP) M_PI) * inv_wavelength;\
        xx = (S_XP - S_XQ) * temp;\
        yy = (S_YP - S_YQ) * temp;\
        rot_angle = ((FP) OMEGA_EARTH) * time_int_sec;\
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;\
        DU = (xx * cos_HA - yy * sin_HA) * rot_angle;\
        DV = temp * sin_Dec;\
        DW = -temp * cos_Dec;}\

/* M_OUT += M */
#define OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(M_OUT, M) {\
        M_OUT.a.x += M.a.x; M_OUT.a.y += M.a.y;\
        M_OUT.b.x += M.b.x; M_OUT.b.y += M.b.y;\
        M_OUT.c.x += M.c.x; M_OUT.c.y += M.c.y;\
        M_OUT.d.x += M.d.x; M_OUT.d.y += M.d.y;}\

/* Construct source brightness matrix (ignoring c, as it's Hermitian). */
#define OSKAR_CONSTRUCT_B(FP, B, SRC_I, SRC_Q, SRC_U, SRC_V) {\
        FP s_I__, s_Q__; s_I__ = SRC_I; s_Q__ = SRC_Q;\
        B.b.x = SRC_U; B.b.y = SRC_V;\
        B.a.x = s_I__ + s_Q__; B.d.x = s_I__ - s_Q__;}\

/* Evaluates 1D linear baseline index for stations P and Q. */
#define OSKAR_BASELINE_INDEX(NUM_STATIONS, P, Q) \
    (Q * (NUM_STATIONS - 1) - (Q - 1) * Q / 2 + P - Q - 1)
