/* Copyright (c) 2019-2025, The OSKAR Developers. See LICENSE file. */

#define OSKAR_EVALUATE_TEC_SCREEN_ARGS(FP) \
        const int isoplanatic,\
        const int num_points, GLOBAL_IN(FP, l), GLOBAL_IN(FP, m),\
        GLOBAL_IN(FP, hor_x), GLOBAL_IN(FP, hor_y), GLOBAL_IN(FP, hor_z),\
        const FP station_u, const FP station_v, const FP inv_frequency_hz,\
        const FP field_x, const FP field_y, const FP field_z,\
        const FP screen_height_m, const FP inv_pixel_size_m,\
        const int screen_num_pixels_x, const int screen_num_pixels_y,\
        GLOBAL_IN(FP, screen), const int offset_out

#define OSKAR_TEC_VALUE(FP) {\
    FP s_l, s_m;\
    if (isoplanatic) {\
        s_l = (FP) 0;\
        s_m = (FP) 0;\
    }\
    else {\
        s_l = l[i];\
        s_m = m[i];\
    }\
    const FP world_x = (station_u + s_l * screen_height_m) * inv_pixel_size_m;\
    const FP world_y = (station_v + s_m * screen_height_m) * inv_pixel_size_m;\
    const int pix_x = screen_num_pixels_x / 2 + ROUND(FP, world_x);\
    const int pix_y = screen_num_pixels_y / 2 + ROUND(FP, world_y);\
    if (pix_x >= 0 && pix_y >= 0 &&\
            pix_x < screen_num_pixels_x && pix_y < screen_num_pixels_y)\
    {\
        tec = screen[pix_x + pix_y * screen_num_pixels_x];\
    }\
}

#define OSKAR_TEC_TO_PHASE(FP, TEC, INV_FREQ_HZ)\
    (TEC * ((FP) -8.44797245e9) * INV_FREQ_HZ)

#define OSKAR_EVALUATE_TEC_SCREEN(NAME, FP, FP2)\
KERNEL(NAME) (OSKAR_EVALUATE_TEC_SCREEN_ARGS(FP), GLOBAL_OUT(FP2, out))\
{\
    (void) field_x;\
    (void) field_y;\
    (void) field_z;\
    (void) hor_x;\
    (void) hor_y;\
    (void) hor_z;\
    KERNEL_LOOP_X(int, i, 0, num_points)\
    FP2 comp;\
    FP tec = (FP) 0;\
    OSKAR_TEC_VALUE(FP)\
    if (tec != (FP) 0) {\
        FP sin_phase, cos_phase;\
        const FP phase = OSKAR_TEC_TO_PHASE(FP, tec, inv_frequency_hz);\
        SINCOS(phase, sin_phase, cos_phase);\
        comp.x = cos_phase; comp.y = sin_phase;\
    } else {\
        comp.x = (FP) 1; comp.y = (FP) 0;\
    }\
    out[i + offset_out] = comp;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)

#define OSKAR_EVALUATE_TEC_SCREEN_WITH_FARADAY_ROTATION(NAME, FP, FP4c)\
KERNEL(NAME) (OSKAR_EVALUATE_TEC_SCREEN_ARGS(FP), GLOBAL_OUT(FP4c, out))\
{\
    KERNEL_LOOP_X(int, i, 0, num_points)\
    FP4c matx;\
    FP tec = (FP) 0;\
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, matx)\
    OSKAR_TEC_VALUE(FP)\
    if (tec != (FP) 0) {\
        FP sin_phase, cos_phase, sin_angle, cos_angle;\
        /* Magnetic field along line of sight to the source. */\
        /* Source vector components are negated because the */\
        /* direction is from the source to the observer. */\
        FP b = -hor_x[i] * field_x - hor_y[i] * field_y - hor_z[i] * field_z;\
        b *= ((FP) 1e-9); /* Convert from nT to Wb/m^2. */\
        /* Use equation 2 in ITU-R P.531-6. */\
        const FP f2 = inv_frequency_hz * inv_frequency_hz;\
        const FP faraday_angle = ((FP) 2.36e2) * b * tec * f2;\
        SINCOS(faraday_angle, sin_angle, cos_angle);\
        const FP phase = OSKAR_TEC_TO_PHASE(FP, tec, inv_frequency_hz);\
        SINCOS(phase, sin_phase, cos_phase);\
        matx.a.x = cos_angle * cos_phase; matx.b.x = -sin_angle * cos_phase;\
        matx.c.x = sin_angle * cos_phase; matx.d.x = cos_angle * cos_phase;\
        matx.a.y = cos_angle * sin_phase; matx.b.y = -sin_angle * sin_phase;\
        matx.c.y = sin_angle * sin_phase; matx.d.y = cos_angle * sin_phase;\
    } else {\
        matx.a.x = (FP) 1; matx.d.x = (FP) 1;\
    }\
    out[i + offset_out] = matx;\
    KERNEL_LOOP_END\
}\
OSKAR_REGISTER_KERNEL(NAME)
