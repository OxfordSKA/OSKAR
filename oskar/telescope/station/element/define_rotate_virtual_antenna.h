/* Copyright (c) 2023-2024, The OSKAR Developers. See LICENSE file. */

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
    const FP2 x_x = jones[j].a, x_y = jones[j].b;\
    const FP2 y_x = jones[j].c, y_y = jones[j].d;\
    jones[j].a.x = cos_angle * x_x.x - sin_angle * y_x.x; /* X_X */\
    jones[j].a.y = cos_angle * x_x.y - sin_angle * y_x.y;\
    jones[j].b.x = cos_angle * x_y.x - sin_angle * y_y.x; /* X_Y */\
    jones[j].b.y = cos_angle * x_y.y - sin_angle * y_y.y;\
    jones[j].c.x = sin_angle * x_x.x + cos_angle * y_x.x; /* Y_X */\
    jones[j].c.y = sin_angle * x_x.y + cos_angle * y_x.y;\
    jones[j].d.x = sin_angle * x_y.x + cos_angle * y_y.x; /* Y_Y */\
    jones[j].d.y = sin_angle * x_y.y + cos_angle * y_y.y;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
