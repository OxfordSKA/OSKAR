/* Copyright (c) 2014-2019, The University of Oxford. See LICENSE file. */

#define OSKAR_GEOMETRIC_DIPOLE(FP, PHI, E_THETA, E_PHI) {\
    SINCOS(PHI, sin_phi, cos_phi);\
    E_THETA.x = cos_theta * cos_phi;\
    E_PHI.x = -sin_phi;\
    E_THETA.y = E_PHI.y = (FP) 0;\
    }\

#define OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN(NAME, FP, FP2)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int stride,\
        const int E_theta_offset,\
        const int E_phi_offset,\
        GLOBAL FP2* E_theta,\
        GLOBAL FP2* E_phi)\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP sin_phi, cos_phi;\
    const int i_out = i * stride;\
    const int theta_out = i_out + E_theta_offset;\
    const int phi_out   = i_out + E_phi_offset;\
    const FP theta_ = theta[i];\
    const FP cos_theta = cos(theta_);\
    const FP phi_ = phi[i];\
    OSKAR_GEOMETRIC_DIPOLE(FP, phi_, E_theta[theta_out], E_phi[phi_out])\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_SCALAR(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int n,\
        GLOBAL_IN(FP, theta),\
        GLOBAL_IN(FP, phi),\
        const int stride,\
        const int offset,\
        GLOBAL_OUT(FP2, pattern))\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    FP amp, sin_phi, cos_phi, phi_;\
    FP4c val;\
    const int i_out = i * stride + offset;\
    const FP theta_ = theta[i];\
    const FP cos_theta = cos(theta_);\
    phi_ = phi[i];\
    OSKAR_GEOMETRIC_DIPOLE(FP, phi_, val.a, val.b)\
    phi_ += ((FP) M_PI / (FP)2);\
    OSKAR_GEOMETRIC_DIPOLE(FP, phi_, val.c, val.d)\
    amp = val.a.x * val.a.x + val.b.x * val.b.x +\
            val.c.x * val.c.x + val.d.x * val.d.x;\
    amp /= (FP)2;\
    amp = sqrt(amp);\
    pattern[i_out].x = amp;\
    pattern[i_out].y = (FP)0;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
