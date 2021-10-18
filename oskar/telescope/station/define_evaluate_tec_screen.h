/* Copyright (c) 2019-2021, The OSKAR Developers. See LICENSE file. */

#define OSKAR_EVALUATE_TEC_SCREEN(NAME, FP, FP2)\
KERNEL(NAME) (const int isoplanatic,\
        const int num_points, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m),\
        const FP station_u, const FP station_v, const FP inv_frequency_hz,\
        const FP screen_height_m, const FP inv_pixel_size_m,\
        const int screen_num_pixels_x, const int screen_num_pixels_y,\
        GLOBAL_IN(FP, screen), const int offset_out, GLOBAL_OUT(FP2, out))\
{\
    const int screen_half_x = screen_num_pixels_x / 2;\
    const int screen_half_y = screen_num_pixels_y / 2;\
    KERNEL_LOOP_X(int, i, 0, num_points)\
    FP2 comp;\
    FP s_l, s_m;\
    if (isoplanatic) {\
        s_l = (FP)0;\
        s_m = (FP)0;\
    }\
    else {\
        s_l = l[i];\
        s_m = m[i];\
    }\
    const FP world_x = (station_u + s_l * screen_height_m) * inv_pixel_size_m;\
    const FP world_y = (station_v + s_m * screen_height_m) * inv_pixel_size_m;\
    const int pix_x = screen_half_x + ROUND(FP, world_x);\
    const int pix_y = screen_half_y + ROUND(FP, world_y);\
    if (pix_x >= 0 && pix_y >= 0 &&\
            pix_x < screen_num_pixels_x && pix_y < screen_num_pixels_y)\
    {\
        FP sin_phase, cos_phase;\
        const FP tec = screen[pix_x + pix_y * screen_num_pixels_x];\
        const FP phase = tec * ((FP) -8.44797245e9) * inv_frequency_hz;\
        SINCOS(phase, sin_phase, cos_phase);\
        comp.x = cos_phase; comp.y = sin_phase;\
    }\
    else\
    {\
        comp.x = (FP)1; comp.y = (FP)0;\
    }\
    out[i + offset_out] = comp;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
