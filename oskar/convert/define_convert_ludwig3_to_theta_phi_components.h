/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI(NAME, FP, FP2) KERNEL(NAME) (\
        const int num, GLOBAL_IN(FP, phi), const int stride, const int off_h,\
        const int off_v, GLOBAL FP2* h_theta, GLOBAL FP2* v_phi)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_p, cos_p;\
    const FP p = phi[i];\
    SINCOS(p, sin_p, cos_p);\
    const int i_h = i * stride + off_h, i_v = i * stride + off_v;\
    const FP2 h = h_theta[i_h], v = v_phi[i_v];\
    h_theta[i_h].x = h.x * cos_p + v.x * sin_p;\
    h_theta[i_h].y = h.y * cos_p + v.y * sin_p;\
    v_phi[i_v].x = -h.x * sin_p + v.x * cos_p;\
    v_phi[i_v].y = -h.y * sin_p + v.y * cos_p;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
