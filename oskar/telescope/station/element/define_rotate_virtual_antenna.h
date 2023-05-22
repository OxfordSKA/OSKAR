/* Copyright (c) 2023, The OSKAR Developers. See LICENSE file. */

#define OSKAR_ROTATE_VIRTUAL_ANTENNA(NAME, FP, FP2, FP4c)\
KERNEL(NAME) (\
        const int n,\
        const int offset,\
        const FP sin_angle,\
        const FP cos_angle,\
        GLOBAL FP4c* jones)\
{\
    KERNEL_LOOP_X(int, i, 0, n)\
    const int j = i + offset;\
    const FP2 x_theta = jones[j].a, x_phi = jones[j].b;\
    const FP2 y_theta = jones[j].c, y_phi = jones[j].d;\
    jones[j].a.x = cos_angle * x_theta.x + sin_angle * y_theta.x; /* X_theta */\
    jones[j].a.y = cos_angle * x_theta.y + sin_angle * y_theta.y;\
    jones[j].b.x = cos_angle * x_phi.x + sin_angle * y_phi.x; /* X_phi */\
    jones[j].b.y = cos_angle * x_phi.y + sin_angle * y_phi.y;\
    jones[j].c.x = -sin_angle * x_theta.x + cos_angle * y_theta.x; /* Y_theta */\
    jones[j].c.y = -sin_angle * x_theta.y + cos_angle * y_theta.y;\
    jones[j].d.x = -sin_angle * x_phi.x + cos_angle * y_phi.x; /* Y_phi */\
    jones[j].d.y = -sin_angle * x_phi.y + cos_angle * y_phi.y;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
