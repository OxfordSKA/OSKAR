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
    const FP2 x_theta_ = jones[j].a, x_phi_ = jones[j].b;\
    const FP2 y_theta_ = jones[j].c, y_phi_ = jones[j].d;\
    jones[j].a.x = x_theta_.x * cos_angle - x_phi_.x * sin_angle;\
    jones[j].a.y = x_theta_.y * cos_angle - x_phi_.y * sin_angle;\
    jones[j].b.x = x_theta_.x * sin_angle + x_phi_.x * cos_angle;\
    jones[j].b.y = x_theta_.y * sin_angle + x_phi_.y * cos_angle;\
    jones[j].c.x = y_theta_.x * cos_angle - y_phi_.x * sin_angle;\
    jones[j].c.y = y_theta_.y * cos_angle - y_phi_.y * sin_angle;\
    jones[j].d.x = y_theta_.x * sin_angle + y_phi_.x * cos_angle;\
    jones[j].d.y = y_theta_.y * sin_angle + y_phi_.y * cos_angle;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
