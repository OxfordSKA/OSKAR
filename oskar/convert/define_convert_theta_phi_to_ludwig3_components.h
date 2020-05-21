/* Copyright (c) 2020, The University of Oxford. See LICENSE file. */

#define OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3(NAME, FP, FP2, FP4c) KERNEL(NAME) (\
        const int num,\
        GLOBAL_IN(FP, phi_x),\
        GLOBAL_IN(FP, phi_y),\
        const int swap_xy,\
        const int offset,\
        GLOBAL FP4c *jones)\
{\
    KERNEL_LOOP_X(int, i, 0, num)\
    FP sin_p_x, cos_p_x, sin_p_y, cos_p_y;\
    FP2 x_theta_, x_phi_, y_theta_, y_phi_;\
    const FP p_x = phi_x[i];\
    const FP p_y = phi_y[i];\
    SINCOS(p_x, sin_p_x, cos_p_x);\
    SINCOS(p_y, sin_p_y, cos_p_y);\
    const int j = i + offset;\
    if (swap_xy) {\
        x_theta_ = jones[j].c, x_phi_ = jones[j].d;\
        y_theta_ = jones[j].a, y_phi_ = jones[j].b;\
    } else {\
        x_theta_ = jones[j].a, x_phi_ = jones[j].b;\
        y_theta_ = jones[j].c, y_phi_ = jones[j].d;\
    }\
    jones[j].a.x = x_theta_.x * cos_p_x - x_phi_.x * sin_p_x;\
    jones[j].a.y = x_theta_.y * cos_p_x - x_phi_.y * sin_p_x;\
    jones[j].b.x = x_theta_.x * sin_p_x + x_phi_.x * cos_p_x;\
    jones[j].b.y = x_theta_.y * sin_p_x + x_phi_.y * cos_p_x;\
    jones[j].c.x = y_theta_.x * sin_p_y + y_phi_.x * cos_p_y;\
    jones[j].c.y = y_theta_.y * sin_p_y + y_phi_.y * cos_p_y;\
    jones[j].d.x = y_theta_.x * cos_p_y - y_phi_.x * sin_p_y;\
    jones[j].d.y = y_theta_.y * cos_p_y - y_phi_.y * sin_p_y;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
